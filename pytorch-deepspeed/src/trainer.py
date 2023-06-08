"""
This is main class file
"""
import argparse
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.plugins.environments import ClusterEnvironment

from src.data_loader import T5DataModule
from src.model_training import T5FineTuner

from metaflow import current
import logging

torch.cuda.empty_cache()
pl.seed_everything(84)

class MetaflowEnvironment(ClusterEnvironment):
    """
    Quick Dirty MF environment in PTL for Single GPU Multi-node training.
    """

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        return current.parallel.main_ip

    @property
    def main_port(self) -> int:
        return 9001 # Fix me

    @staticmethod
    def detect() -> bool:
        return True

    def world_size(self) -> int:
        return int(current.parallel.num_nodes)

    def set_world_size(self, size: int) -> None:
        logging.debug("MetaflowEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        return int(current.parallel.node_index)

    def set_global_rank(self, rank: int) -> None:
        logging.debug("MetaflowEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        return 0

    def node_rank(self) -> int:
        return self.global_rank()


class T5FineTune:
    """
    This class is using for fine-tune T5 based models
    """

    def __init__(self, model_type="t5", model_name="t5-base") -> None:
        """ Initiates T5FineTune class and loads T5, MT5, ByT5, t0, or flan-t5 model for fine-tuning """

        if model_type in ["t5", "flan-t5"]:
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "mt5":
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = MT5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "byt5":
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_name}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                f"{model_name}", return_dict=True
            )
        elif model_type == "t0":
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                f"{model_name}", return_dict=True
            )

    def train(
            self,
            train_df: pd.DataFrame,
            eval_df: pd.DataFrame,
            args: argparse.Namespace = argparse.Namespace(),):
        """

        :param train_df: Dataframe must have 2 column --> "source_text" and "target_text":
        :param eval_df: Dataframe must have 2 column --> "source_text" and "target_text":
        :param args: arguments
        :return: trained model
        """
        self.data_module = T5DataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=args.batch_size,
            source_max_token_length=args.source_max_token_length,
            target_max_token_length=args.target_max_token_length,
            num_workers=args.dataloader_num_workers)

        self.t5_model = T5FineTuner(args, tokenizer=self.tokenizer, model=self.model)

        callbacks = [TQDMProgressBar(refresh_rate=1)]

        if args.early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                                patience=args.early_stopping_patience_epochs, verbose=True, mode="min")
            callbacks.append(early_stop_callback)

        gpus = args.n_gpu if args.use_gpu else 0

        # add logger
        loggers = True if args.logger == "default" else args.logger
        
        env = MetaflowEnvironment()
        
        deepspeed = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
            process_group_backend="nccl",
            cluster_environment=env,
            min_loss_scale=1
        )
        
        # prepare trainer
        trainer = pl.Trainer(
            logger=loggers, 
            callbacks=callbacks, 
            max_epochs=args.max_epochs, 
            accelerator = "gpu" if args.use_gpu else "cpu",
            num_nodes=args.num_nodes,
            strategy=deepspeed,
            devices=gpus,
            precision=args.precision, 
            log_every_n_steps=1,
            deterministic=True,
            enable_checkpointing=True,
            enable_model_summary=True
        )

        # fit trainer
        trainer.fit(self.t5_model, self.data_module)

    def load_model(self, model_type: str = "t5", model_dir: str = "outputs", use_gpu: bool = False):
        """
        This function is using for load trained models
        :param model_type: model type
        :param model_dir: trained model directory
        :param use_gpu: gpu usage
        :return: loaded model
        """
        if model_type in ["t5", "flan-t5"]:
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = T5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "mt5":
            self.model = MT5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = MT5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "byt5":
            self.model = T5ForConditionalGeneration.from_pretrained(f"{model_dir}")
            self.tokenizer = ByT5Tokenizer.from_pretrained(f"{model_dir}")
        elif model_type == "t0":
            self.model = AutoModelForSeq2SeqLM.from_pretrained(f"{model_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}")

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise "exception ---> no gpu found. set use_gpu=False, to use CPU"
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

    def predict(
            self,
            source_text: str,
            max_length: int = 512,
            num_return_sequences: int = 1,
            num_beams: int = 2,
            top_k: int = 10,
            top_p: float = 0.95,
            do_sample: bool = True,
            repetition_penalty: float = 2.5,
            length_penalty: float = 1.0,
            early_stopping: bool = True,
            skip_special_tokens: bool = True,
            clean_up_tokenization_spaces: bool = True,
    ):

        input_ids = self.tokenizer.encode(
            source_text, return_tensors="pt", add_special_tokens=True
        )
        input_ids = input_ids.to(self.device)
        generated_ids = self.model.generate(
            input_ids=input_ids,
            num_beams=num_beams,
            max_length=max_length,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=num_return_sequences,
        )
        preds = [
            self.tokenizer.decode(
                g,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            )
            for g in generated_ids
        ]
        return preds