"""
This is main class file
"""
import argparse
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    PreTrainedTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from pytorch_lightning.plugins.environments import ClusterEnvironment
import torch.distributed as dist

from src.data_loader import MistralDataModule
from src.model_training import MistralFineTuner
from src.consts import *

from metaflow import current
import logging
import os
from tqdm.contrib.logging import tqdm_logging_redirect
import pandas as pd

torch.cuda.empty_cache()
pl.seed_everything(84)


class MetaflowEnvironment(ClusterEnvironment):
    """
    Quick Dirty MF environment in PTL for Multi-GPU Multi-node training.
    """

    @property
    def creates_processes_externally(self) -> bool:
        return True

    @property
    def main_address(self) -> str:
        return current.parallel.main_ip

    @property
    def main_port(self) -> int:
        return 9001  # Fix me

    @staticmethod
    def detect() -> bool:
        return True

    def world_size(self) -> int:
        return int(current.parallel.num_nodes) * int(N_GPU)

    def set_world_size(self, size: int) -> None:
        logging.debug("MetaflowEnvironment.set_world_size was called, but setting world size is not allowed. Ignored.")

    def global_rank(self) -> int:
        return int(current.parallel.node_index) * int(N_GPU) + int(os.environ.get("LOCAL_RANK", 0))

    def set_global_rank(self, rank: int) -> None:
        logging.debug(
            "MetaflowEnvironment.set_global_rank was called, but setting global rank is not allowed. Ignored.")

    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", 0))

    def node_rank(self) -> int:
        return int(current.parallel.node_index)


class MistralFineTune:
    """
    This class is using for fine-tune Mistral models
    """

    def __init__(self, model_name) -> None:
        """ Initiates MistralFineTune class and loads Mistral model for fine-tuning """
        MAX_RETRIES = 5
        for retry in range(MAX_RETRIES):
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.bfloat16,
                    device_map=None,
                    use_cache=True,
                    resume_download=True,
                    trust_remote_code=True,
                    cache_dir=".cache/mistral",
                    return_dict=True
                )
                break
            except Exception as e:
                print(f"Attempt {retry + 1}/{MAX_RETRIES} failed with error: {e}")
                if retry < MAX_RETRIES - 1:
                    print("Retrying...")
                else:
                    print("Max retries reached. Exiting.")

        self.model.config.use_cache = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code="true", padding_side="left"
        )

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def train(
            self,
            output_dir: str,
            source_max_token_length: int,
            target_max_token_length: int,
            batch_size: int,
            max_epochs: int,
            learning_rate: float,
            weight_decay: float,
            adam_epsilon: float,
            warmup_steps: int,
            gradient_accumulation_steps: int,
            n_gpu: int,
            num_nodes: int,
            early_stopping_patience_epochs: int,
            precision: str,
            logger: str,
            dataloader_num_workers: int,
            opt_level: str,
            max_grad_norm: float,
            seed: int,
            early_stop_callback: bool,
            save_only_last_epoch: bool,
            fp_16: bool,
            use_gpu: bool
    ):
        train_df = pd.read_csv("train.csv")
        eval_df = pd.read_csv("val.csv")

        self.data_module = MistralDataModule(
            train_df,
            eval_df,
            self.tokenizer,
            batch_size=batch_size,
            source_max_token_length=source_max_token_length,
            target_max_token_length=target_max_token_length,
            num_workers=dataloader_num_workers
        )

        args_dict = dict(
            output_dir=output_dir,
            source_max_token_length=source_max_token_length,
            target_max_token_length=target_max_token_length,
            batch_size=batch_size,
            max_epochs=max_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_epsilon=adam_epsilon,
            warmup_steps=warmup_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_gpu=use_gpu,
            n_gpu=n_gpu,
            num_nodes=num_nodes,
            early_stop_callback=early_stop_callback,
            early_stopping_patience_epochs=early_stopping_patience_epochs,
            precision=precision,
            logger=logger,
            dataloader_num_workers=dataloader_num_workers,
            save_only_last_epoch=save_only_last_epoch,
            fp_16=fp_16,
            opt_level=opt_level,
            max_grad_norm=max_grad_norm,
            seed=seed
        )
        args = argparse.Namespace(**args_dict)

        self.mistral_model = MistralFineTuner(args, tokenizer=self.tokenizer, model=self.model)

        callbacks = [TQDMProgressBar()]

        if early_stopping_patience_epochs > 0:
            early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00,
                                                patience=early_stopping_patience_epochs, verbose=True, mode="min")
            callbacks.append(early_stop_callback)

        gpus = n_gpu if use_gpu else 0

        # add logger
        loggers = True if logger == "default" else logger

        env = MetaflowEnvironment()

        deepspeed = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            allgather_bucket_size=5e8,
            reduce_bucket_size=5e8,
            process_group_backend="nccl",
            cluster_environment=env,
            min_loss_scale=1,
            cpu_checkpointing=True
        )

        # prepare trainer
        trainer = pl.Trainer(
            logger=loggers,
            callbacks=callbacks,
            max_epochs=max_epochs,
            accelerator="gpu" if use_gpu else "cpu",
            num_nodes=num_nodes,
            strategy=deepspeed,
            devices=gpus,
            precision=precision,
            log_every_n_steps=10,
            deterministic=True,
            enable_checkpointing=True,
            enable_model_summary=True
        )

        with tqdm_logging_redirect():
            # fit trainer
            trainer.fit(self.mistral_model, self.data_module)

    def load_model(self, model_dir: str = "outputs", use_gpu: bool = False):
        """
        This function is using for load trained models
        :param model_type: model type
        :param model_dir: trained model directory
        :param use_gpu: gpu usage
        :return: loaded model
        """

        self.model = AutoModelForCausalLM.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

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
