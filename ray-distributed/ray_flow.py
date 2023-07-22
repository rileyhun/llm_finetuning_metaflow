import logging
import sys
import os
from metaflow import (
    ray_parallel,
    FlowSpec,
    step,
    batch,
    current,
    conda_base,
    environment,
    IncludeFile,
    Parameter
)
from custom_decorators import pip, enable_decorator
from gpu_profile import gpu_profile
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
        "conda-forge::matplotlib": "3.5.3",
        "conda-forge::pandas": "1.5.3",
        "conda-forge::pytorch-lightning": "1.8.6",
        "conda-forge::torchmetrics": "0.11.4"
    },
    python="3.10.4",
)
class RayFlow(FlowSpec):
    num_parallel = Parameter(
        "num_parallel", help="Number of nodes in cluster", default=4
    )

    @step
    def start(self):
        self.next(self.train, num_parallel=self.num_parallel)

    @gpu_profile(interval=1)
    @environment(
        vars={
            "EN_BATCH": os.getenv("EN_BATCH")
        }
    )
    @enable_decorator(
        batch(gpu=N_GPU, cpu=N_CPU, memory=MEMORY, queue=QUEUE_NAME), flag=os.getenv("EN_BATCH")
    )
    @ray_parallel(master_port=6379)
    @step
    def train(self):
        import os
        import numpy as np
        import pytorch_lightning as pl
        from pytorch_lightning import trainer
        from pytorch_lightning.loggers.csv_logs import CSVLogger
        from pytorch_lightning.callbacks import ModelCheckpoint
        from ray.air.config import RunConfig, ScalingConfig, CheckpointConfig
        from ray.train.lightning import (
            LightningTrainer,
            LightningConfigBuilder,
            LightningCheckpoint,
        )

        from model_training import MNISTClassifier
        from data_loader import MNISTDataModule
        import ray

        if current.parallel.node_index == 0:
            print(f"{current.parallel.main_ip}:6379")
            context = ray.init("auto")
            print(context.dashboard_url)

            datamodule = MNISTDataModule(batch_size=128)

            def build_lightning_config_from_existing_code(use_gpu):
                config_builder = LightningConfigBuilder()
                config_builder.module(cls=MNISTClassifier, lr=1e-3, feature_dim=128)
                config_builder.checkpointing(monitor="val_accuracy", mode="max", save_top_k=3)
                config_builder.trainer(
                    max_epochs=10,
                    accelerator="gpu" if use_gpu else "cpu",
                    log_every_n_steps=100,
                    logger=CSVLogger("logs"),
                )
                config_builder.fit_params(datamodule=datamodule)
                lightning_config = config_builder.build()
                return lightning_config

            use_gpu = True  # Set it to False if you want to run without GPUs
            lightning_config = build_lightning_config_from_existing_code(use_gpu=use_gpu)
            scaling_config = ScalingConfig(num_workers=self.num_parallel, use_gpu=use_gpu)

            run_config = RunConfig(
                name="ptl-mnist-example",
                storage_path="/tmp/ray_results",
                checkpoint_config=CheckpointConfig(
                    num_to_keep=3,
                    checkpoint_score_attribute="val_accuracy",
                    checkpoint_score_order="max",
                ),
            )

            trainer = LightningTrainer(
                lightning_config=lightning_config,
                scaling_config=scaling_config,
                run_config=run_config,
            )

            result = trainer.fit()
            print("Validation Accuracy: ", result.metrics["val_accuracy"])
            print(result)

        self.next(self.multinode_end)

    @step
    def multinode_end(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayFlow()