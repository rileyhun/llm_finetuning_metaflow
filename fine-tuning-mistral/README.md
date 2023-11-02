# Fine-Tuning Mistral-7B with PyTorch Lightning & Deepspeed

### Description
Example of multi-node multi-GPU distributed fine-tuning of Mistral-7B model, which surpasses Llama-2 in most benchmarking results. 
Uses PyTorch Lightning with Deepspeed integration, and pretrained model from Hugging Face hub. 
Showcases how to leverage `@parallel` decorator to perform distributed training.

### How to Run Example

1. Use `PyTorch 2.0.0 Python 3.10 CPU Optimized Image` + `Python 3` Kernel

2. Install metaflow + experimental bleeding edge conda decorator
```bash
pip install metaflow
pip install git+https://github.com/Netflix/metaflow-nflx-extensions
pip install python-dotenv
```

3. Install mamba (speeds up resolving environment)
```bash
conda install mamba
```

5. Run Metaflow flow
```bash
export METAFLOW_CONDA_DEPENDENCY_RESOLVER=mamba
export CONDA_CHANNEL_PRIORITY=flexible

source ~/.bashrc

python python mistral_fine_tune.py --environment=conda run
```
