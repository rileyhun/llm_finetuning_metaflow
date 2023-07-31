import logging
import sys
import os
from metaflow import (
    ray_parallel,
    parallel,
    FlowSpec,
    step,
    batch,
    current,
    conda_base,
    environment,
    IncludeFile,
    Parameter,
    Task
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
        "conda-forge::deepspeed": "0.9.2"
    },
    python="3.10.4",
)
class RayDeepspeedFlow(FlowSpec):
    num_workers = Parameter(
        "num_workers", help="Number of nodes in cluster", default=12
    )

    batch_size = Parameter(
        "batch_size", help="Batch size", default=16
    )

    epochs = Parameter(
        "epochs", help="Number of epochs", default=1
    )

    storage_path = Parameter(
        "storage_path", help="Storage Path", default="s3://ampstrn-c-uw2-s3-amp/checkpoint/ray-deepspeed"
    )

    use_gpu = Parameter(
        "use_gpu", help="Use GPU?", default=True
    )

    @step
    def start(self):

        self.previous_run_id = current.run_id
        self.previous_task_id = current.task_id
        self.previous_step_name = current.step_name

        self.next(self.train, num_parallel=self.num_workers)

    @gpu_profile(interval=1)
    @environment(
        vars={
            "EN_BATCH": os.getenv("EN_BATCH"),
            "RAY_AIR_REENABLE_DEPRECATED_SYNC_TO_HEAD_NODE": "1",
            "RAY_health_check_period_ms": "1000000000",
            "RAY_BACKEND_LOG_LEVEL": "debug",
            "RAY_DATA_STRICT_MODE": "0",
            "NVIDIA_DRIVER_CAPABILITIES": "compute,utility",
            "CUDA_HOME": "/usr/local/cuda",
            "NCCL_DEBUG": "INFO",
            "NCCL_SOCKET_IFNAME": "eth0"
        }
    )
    @enable_decorator(
        batch(gpu=N_GPU, cpu=N_CPU, memory=MEMORY, queue=QUEUE_NAME, shared_memory=12000), flag=os.getenv("EN_BATCH")
    )
    @pip(libraries={
        "opencv_python_headless": "4.5.5.62",
        "transformers": "4.26.0",
        "datasets": "2.10.1",
        "evaluate": "",
        "accelerate": "0.16.0",
        "ninja": "1.11.1",
        "scikit-learn": ""
    })
    @ray_parallel(master_port=6379)
    @step
    def train(self):
        import ray.data
        from ray.data.preprocessors import BatchMapper
        from ray.train.huggingface import TransformersTrainer
        from ray.air import RunConfig, ScalingConfig
        from ray.data.preprocessors import Chain
        from datasets import load_dataset
        from data_loader import split_text, tokenize
        from trainer import trainer_init_per_worker
        import ray
        import time

        if current.parallel.node_index == 0:

            ray.init(
                runtime_env={
                    "pip": [
                        "datasets==2.10.1",
                        "evaluate",
                        "scikit-learn",
                        "accelerate==0.16.0",
                        "transformers==4.26.0",
                        "torch==1.12.0",
                        "deepspeed==0.9.2",
                        "ninja==1.11.1"
                    ]
                }
            )
            print(ray.cluster_resources())

            print("Loading tiny_shakespeare dataset")
            current_dataset = load_dataset("tiny_shakespeare")
            ray_datasets = ray.data.from_huggingface(current_dataset)
            splitter = BatchMapper(split_text, batch_format="pandas")
            tokenizer = BatchMapper(tokenize, batch_format="pandas")

            trainer = TransformersTrainer(
                trainer_init_per_worker=trainer_init_per_worker,
                trainer_init_config={
                    "batch_size": self.batch_size,  # per device
                    "epochs": self.epochs,
                },
                scaling_config=ScalingConfig(
                    num_workers=self.num_workers,
                    use_gpu=self.use_gpu
                    # resources_per_worker={"GPU": N_GPU, "CPU": int(N_CPU)-1}
                ),
                datasets={"train": ray_datasets["train"], "evaluation": ray_datasets["validation"]},
                preprocessor=Chain(splitter, tokenizer),
                run_config=RunConfig(storage_path=self.storage_path),
            )

            results = trainer.fit()

            print(results)

            checkpoint = results.checkpoint
            print(checkpoint)
        else:
            while not Task(
                    f"{current.flow_name}/{self.previous_run_id}/{current.step_name}/control-{current.run_id}-{self.previous_step_name}-{self.previous_task_id}").finished:
                time.sleep(10)

        self.next(self.multinode_end)

    @step
    def multinode_end(self, inputs):
        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    RayDeepspeedFlow()