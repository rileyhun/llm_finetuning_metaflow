import logging
import sys
import os
from metaflow import (
    FlowSpec,
    step,
    batch,
    current,
    pytorch_parallel,
    Parameter,
    conda_base,
    environment,
    IncludeFile,
)
from gpu_profile import gpu_profile
from custom_decorators import pip, enable_decorator
from consts import *

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

try:
    from dotenv import load_dotenv

    load_dotenv(verbose=True, dotenv_path="local.env")
except:
    print("No dot env!")


def create_instruction_prompt(text, all_labels):
    prompt = f""" Classify the following messages into one of the following categories: {','.join(all_labels)}

Message: {text}

Category:"""
    return prompt


def create_raw_prompt(text):
    prompt = f"""{text} /n/n###/n/n"""
    return prompt


@conda_base(
    libraries={
        "pytorch::pytorch": "1.12.0",
        "pytorch::torchvision": "0.13.0",
        "conda-forge::cudatoolkit": "11.3.1",
        "conda-forge::matplotlib": "3.5.3",
        "conda-forge::pandas": "1.5.3",
        "conda-forge::pytorch-lightning": "1.8.6",
        "conda-forge::accelerate": "0.18.0"
    },
    python="3.10.4",
)
class PytorchLightningGptJFineTune(FlowSpec):
    """
    Flow to test @pytorch_parallel and run distributed training
    """

    num_parallel = Parameter(
        "num_parallel", help="Number of nodes in cluster", default=16
    )

    num_gpus = Parameter(
        "num_gpus", help="Number of GPUs per node", default=1)

    num_epochs = Parameter(
        "num_epochs", help="Number of training epochs", default=1
    )

    batch_size = Parameter(
        "batch_size", help="Batch size", default=6
    )

    learning_rate = Parameter(
        "learning_rate", help="Learning rate", default=1e-4
    )

    train_data = IncludeFile("train_data", default="data/train.csv")

    validation_data = IncludeFile("validation_data", default="data/val.csv")

    @step
    def start(self):
        self.next(self.load_data)

    @step
    def load_data(self):
        from io import StringIO
        import pandas as pd

        train_df = pd.read_csv(StringIO(self.train_data))
        val_df = pd.read_csv(StringIO(self.validation_data))

        train_df["completion"] = train_df["label"].apply(lambda x: " " + x)
        val_df["completion"] = val_df["label"].apply(lambda x: " " + x)
        all_labels = set(train_df["completion"].unique())
        train_df["prompt"] = train_df["text"].apply(
            lambda x: create_instruction_prompt(x, all_labels)
        )
        val_df["prompt"] = val_df["text"].apply(
            lambda x: create_instruction_prompt(x, all_labels)
        )

        self.train_df = train_df
        self.val_df = val_df

        self.next(self.train, num_parallel=self.num_parallel)

    @gpu_profile(interval=1)
    @pip(libraries={"opencv_python_headless": "4.5.5.62", "bitsandbytes-cuda113": "0.26.0.post2",
                    "transformers": "4.25.1"})
    @environment(
        vars={
            "EN_BATCH": os.getenv("EN_BATCH"),
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_HOME": "/usr/local/cuda",
            "TOKENIZERS_PARALLELISM": "true",
            "NCCL_DEBUG": "INFO"
        }
    )
    @enable_decorator(
        batch(gpu=N_GPU, cpu=N_CPU, memory=MEMORY, queue=QUEUE_NAME), flag=os.getenv("EN_BATCH")
    )
    @pytorch_parallel
    @step
    def train(self):
        """
        Run a simple torch parallel program where each node creates a 3 x 3 tensor
        with each entry equaling their rank + 1. Then, all reduce is called to sum the
        tensors up.
        """
        # from patches import patch_bits_and_bytes
        # patch_bits_and_bytes(113)

        import torch
        import transformers
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import TQDMProgressBar
        from pytorch_lightning.strategies import DDPFullyShardedNativeStrategy, DDPStrategy
        from pytorch_lightning.plugins.environments import LightningEnvironment
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
        import pandas as pd
        from functools import partial
        import logging
        import os
        from tqdm.contrib.logging import tqdm_logging_redirect

        from custom_datasets import PromptDataset
        from gpt_quant_modules import GPTJBlock, GPTJForCausalLM
        from gpt_finetuner import FinetunerConfig, GPTJ8bitFineTuner
        from finetuning_utils import add_all_adapters, add_attention_adapters

        transformers.models.gptj.modeling_gptj.GPTJBlock = GPTJBlock

        print(f"cuda is available: {torch.cuda.is_available()}")
        print("torch version: ", torch.__version__)
        print("torch cuda version: ", torch.version.cuda)
        print("torch cudnn version: ", torch.backends.cudnn.version())
        print(os.system('nvidia-smi'))

        env = LightningEnvironment()
        env.world_size = lambda: int(current.parallel.node_index)
        env.global_rank = lambda: int(current.parallel.num_nodes)

        logging.info("Create Data Loaders ...")
        tokenizer = transformers.AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_MODEL)
        tokenizer.pad_token = tokenizer.eos_token

        max_prompt_size = (
                int(
                    pd.Series(
                        len(tokenizer.tokenize(e))
                        for e in (
                                self.train_df["prompt"] + " " + self.train_df["completion"]
                        )
                    ).quantile(0.99)
                )
                + 1
        )

        train_dataset = PromptDataset(
            self.train_df, tokenizer, max_prompt_len=max_prompt_size
        )
        val_dataset = PromptDataset(
            self.val_df, tokenizer, max_prompt_len=max_prompt_size
        )

        pl.seed_everything(100)

        logging.info("Create Fine Tuner ...")
        config = FinetunerConfig(
            lr=self.learning_rate, batch_size=self.batch_size, num_epochs=self.num_epochs, adapter_dim=2,
            classification=True
        )
        model_post_init_func = partial(add_all_adapters, adapter_dim=config.adapter_dim)
        finetuner = GPTJ8bitFineTuner(
            model_name=DEFAULT_FINETUNER_MODEL,
            model_post_init_func=model_post_init_func,
            fine_tuning_config=config,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
        )

        logging.info("Training ...")

        with tqdm_logging_redirect():
            fsdp_native = DDPFullyShardedNativeStrategy(
                cpu_offload=CPUOffload(offload_params=True),
                cluster_environment=env,
                process_group_backend="nccl"
            )
            #             ddp = DDPStrategy(
            #               find_unused_parameters=True,
            #               cluster_environment=env,
            #               process_group_backend="gloo",
            #               accelerator="gpu"
            #             )

            trainer = pl.Trainer(
                log_every_n_steps=1,
                devices=self.num_gpus,
                num_nodes=self.num_parallel,
                max_epochs=config.num_epochs,
                deterministic=True,
                enable_checkpointing=True,
                enable_model_summary=True,
                profiler="simple",
                precision=16,
                callbacks=[TQDMProgressBar(refresh_rate=0)],
                strategy=fsdp_native,
                accelerator="gpu"
            )
            trainer.fit(finetuner)

        self.next(self.multinode_end)

    @step
    def multinode_end(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    PytorchLightningGptJFineTune()
