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
    retry
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
        "conda-forge::deepspeed": "0.9.2"
    },
    python="3.10.4",
)
class DeepSpeedFlow(FlowSpec):
    """
    Flow to run distributed fine-tuning!
    """
    
    train_data = IncludeFile("train_data", default="data/train.csv")
    
    test_data = IncludeFile("test_data", default="data/test.csv")
    
    val_data = IncludeFile("val_data", default="data/val.csv")

    num_parallel = Parameter(
        "num_parallel", help="Number of nodes in cluster", default=6
    )
    
    output_dir = Parameter(
        "output_dir", help="Local path to save checkpoints", default="outputs"
    )
    
    source_max_token_length = Parameter(
        "source_max_token_length", default=256
    )
    
    target_max_token_length = Parameter(
        "target_max_token_length", default=256
    )
    
    batch_size = Parameter(
        "batch_size", default=4
    )
    
    max_epochs = Parameter(
        "max_epochs", default=1
    )
    
    learning_rate = Parameter(
        "learning_rate", default=3e-4
    )
    
    weight_decay = Parameter(
        "weight_decay", default=0.1
    )
    
    adam_epsilon = Parameter(
        "adam_epsilon", default=9e-7
    )
    
    warmup_steps = Parameter(
        "warmup_steps", default=0
    )
    
    gradient_accumulation_steps = Parameter(
        "gradient_accumulation_steps", default=16
    )
    
    use_gpu = Parameter(
        "use_gpu", default=True
    )
    
    n_gpu = Parameter(
        "n_gpu", default=1
    )
    
    early_stop_callback = Parameter(
        "early_stop_callback", default=False
    )
    
    early_stopping_patience_epochs = Parameter(
        "early_stopping_patience_epochs", default=0
    )
    
    precision = Parameter(
        "precision", default="bf16"
    )
    
    logger = Parameter(
        "logger", default="default"
    )
    
    dataloader_num_workers = Parameter(
        "dataloader_num_workers", default=2
    )
    
    save_only_last_epoch = Parameter(
        "save_only_last_epoch", default=False
    )
    
    fp_16 = Parameter(
        "fp_16", default=False
    )
    
    opt_level = Parameter(
        "opt_level", default="01"
    )
    
    max_grad_norm = Parameter(
        "max_grad_norm", default=0.5
    )
    
    seed = Parameter(
        "seed", default=42
    )

    @step
    def start(self):
        self.next(self.load_data)
        
    @step
    def load_data(self):
        from io import StringIO
        import pandas as pd
        
        train_df = pd.read_csv(StringIO(self.train_data))
        test_df = pd.read_csv(StringIO(self.test_data))
        val_df = pd.read_csv(StringIO(self.val_data))
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
           
        self.next(self.train, num_parallel=self.num_parallel)
        
    @gpu_profile(interval=1)
    @pip(libraries={"opencv_python_headless": "4.5.5.62", "transformers": "4.25.1", "ninja": "1.11.1"})
    @environment(
        vars={
            "EN_BATCH": os.getenv("EN_BATCH"),
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_HOME": "/usr/local/cuda",
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0",
            "NCCL_P2P_DISABLE": "1",
            "TOKENIZERS_PARALLELISM": "true"
        }
    )
    @enable_decorator(
        batch(gpu=N_GPU, cpu=N_CPU, memory=MEMORY, queue=QUEUE_NAME), flag=os.getenv("EN_BATCH")
    )
    @pytorch_parallel(master_port=9001)
    @step
    def train(self):
        """
        Run fine-tuning on news summary data
        """
        import argparse
        import logging
        import pandas as pd
        import time
        import torch
        from pytorch_lightning.plugins.environments import ClusterEnvironment
        from tqdm.contrib.logging import tqdm_logging_redirect
        from src.trainer import T5FineTune
        
        print("Torch CUDA available?", torch.cuda.is_available(), flush=True)

        model = T5FineTune("t0", "bigscience/T0_3B")
    
        start = time.time()
        args_dict = dict(
            output_dir=self.output_dir,  # path to save the checkpoints
            source_max_token_length=self.source_max_token_length,
            target_max_token_length=self.target_max_token_length,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            adam_epsilon=self.adam_epsilon,
            warmup_steps=self.warmup_steps,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            use_gpu=self.use_gpu,
            n_gpu=self.n_gpu,
            num_nodes=self.num_parallel,
            early_stop_callback=self.early_stop_callback,
            early_stopping_patience_epochs=self.early_stopping_patience_epochs,
            precision=self.precision,
            logger=self.logger,
            dataloader_num_workers=self.dataloader_num_workers,
            save_only_last_epoch=self.save_only_last_epoch,
            fp_16=self.fp_16,  
            opt_level=self.opt_level,
            max_grad_norm=self.max_grad_norm,  
            seed=self.seed
        )
        arguments = argparse.Namespace(**args_dict)
        with tqdm_logging_redirect():
            model.train(
                train_df=self.train_df, 
                eval_df=self.val_df, 
                args=arguments
            )

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
    DeepSpeedFlow()
