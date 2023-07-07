import logging
import sys
import os
from metaflow import (
    FlowSpec,
    step,
    batch,
    current,
    parallel,
    Parameter,
    conda_base,
    environment,
    IncludeFile,
    retry,
)
from gpu_profile import gpu_profile
from custom_decorators import pip, enable_decorator, magicdir
from src.consts import *

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

try:
    from dotenv import load_dotenv

    load_dotenv(verbose=True, dotenv_path="local.env")
except:
    print("No dot env!")


@conda_base(
    libraries={
        "pytorch::pytorch": "1.12.0",
        "pytorch::torchvision": "0.13.0",
        # "conda-forge::cudatoolkit": "11.3.1",
        "conda-forge::matplotlib": "3.5.3",
        "conda-forge::sentencepiece": "0.1.97",
        "conda-forge::pandas": "1.5.3",
        "conda-forge::pytorch-lightning": "1.8.6",
        "conda-forge::scikit-learn": "1.2.2",
        "conda-forge::deepspeed": "0.9.5",
    },
    python="3.10.4",
)
class T5DeepspeedFlow(FlowSpec):
    """
    Flow to run distributed fine-tuning!
    """

    train_data = IncludeFile("train_data", default="data/train.csv")

    test_data = IncludeFile("test_data", default="data/test.csv")

    val_data = IncludeFile("val_data", default="data/val.csv")

    num_nodes = Parameter("num_nodes", help="Number of nodes in cluster", default=4)

    output_dir = Parameter(
        "output_dir", help="Local path to save checkpoints", default="outputs"
    )

    source_max_token_length = Parameter("source_max_token_length", default="256")

    target_max_token_length = Parameter("target_max_token_length", default="256")

    batch_size = Parameter("batch_size", default="12")

    max_epochs = Parameter("max_epochs", default="1")

    learning_rate = Parameter("learning_rate", default="3e-4")

    weight_decay = Parameter("weight_decay", default="0.1")

    adam_epsilon = Parameter("adam_epsilon", default="9e-7")

    warmup_steps = Parameter("warmup_steps", default="0")

    gradient_accumulation_steps = Parameter("gradient_accumulation_steps", default="16")

    n_gpu = Parameter("n_gpu", default="4")

    early_stopping_patience_epochs = Parameter(
        "early_stopping_patience_epochs", default="0"
    )

    precision = Parameter("precision", default="bf16")

    logger = Parameter("logger", default="default")

    dataloader_num_workers = Parameter("dataloader_num_workers", default="4")

    opt_level = Parameter("opt_level", default="01")

    max_grad_norm = Parameter("max_grad_norm", default="0.5")

    seed = Parameter("seed", default="42")

    save_only_last_epoch = Parameter("save_only_last_epoch", default=False)

    use_gpu = Parameter("use_gpu", default=True)

    early_stop_callback = Parameter("early_stop_callback", default=False)

    fp_16 = Parameter("fp_16", default=False)

    @step
    def start(self):
        self.next(self.train, num_parallel=self.num_nodes)
    
    @gpu_profile(interval=1)
    @pip(
        libraries={
            "opencv_python_headless": "4.5.5.62",
            "transformers": "4.25.1",
            "ninja": "1.11.1",
            "click": "8.0.4",
            "pydantic": "1.10.11"
        }
    )
    @environment(
        vars={
            "EN_BATCH": os.getenv("EN_BATCH"),
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_HOME": "/usr/local/cuda",
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_SHM_DISABLE": "1",
            "TOKENIZERS_PARALLELISM": "true"
        }
    )
    @enable_decorator(
        batch(gpu=N_GPU, cpu=N_CPU, memory=MEMORY, queue=QUEUE_NAME),
        flag=os.getenv("EN_BATCH"),
    )
    @parallel
    @step
    def train(self):
        """
        Run fine-tuning on news summary data
        """
        import time
        import torch
        import subprocess
        import pandas as pd
        from io import StringIO

        print("Torch CUDA available?", torch.cuda.is_available(), flush=True)
        
        print("Reading in data and save to csv ...")
        
        train_df = pd.read_csv(StringIO(self.train_data)).to_csv("train.csv", index=False)
        val_df = pd.read_csv(StringIO(self.val_data)).to_csv("val.csv", index=False)
                               
        start = time.time()

        print("Calling PTL process...\n\n")
        
        subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={str(self.n_gpu)}",
                f"--nnodes={str(self.num_nodes)}",
                f"--rdzv_id=metaflow_{current.run_id}",
                "--rdzv_backend=c10d",
                f"--rdzv_endpoint={current.parallel.main_ip}:29400",
                "deepspeed_trainer.py",
                "--output-dir", self.output_dir,
                "--source-max-token-length", self.source_max_token_length,
                "--target-max-token-length", self.target_max_token_length,
                "--batch-size", self.batch_size,
                "--max-epochs", self.max_epochs,
                "--learning-rate", self.learning_rate,
                "--weight-decay", self.weight_decay,
                "--adam-epsilon", self.adam_epsilon,
                "--warmup-steps", self.warmup_steps,
                "--gradient-accumulation-steps", self.gradient_accumulation_steps,
                "--n-gpu", self.n_gpu,
                "--num-nodes=%d" % self.num_nodes,
                "--early-stopping-patience-epochs", self.early_stopping_patience_epochs,
                "--precision", self.precision,
                "--logger", self.logger,
                "--dataloader-num-workers", self.dataloader_num_workers,
                "--opt-level", self.opt_level,
                "--max-grad-norm", self.max_grad_norm,
                "--seed", self.seed,
            ]
            + (["--early-stop-callback"] if self.early_stop_callback else [])
            + (["--save-only-last-epoch"] if self.save_only_last_epoch else [])
            + (["--fp-16"] if self.fp_16 else [])
            + (["--use-gpu"] if self.use_gpu else []),
            check=True,
        )
        print("PTL process completed!")

        end = time.time()
        elapsed = end - start
        print(f"Time elapsed {elapsed/60:.2f} min")

        self.next(self.multinode_end)

    @step
    def multinode_end(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    T5DeepspeedFlow()
