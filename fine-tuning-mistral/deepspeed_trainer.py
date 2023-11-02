import click
import logging
from metaflow import current
from src.trainer import MistralFineTune
from src.consts import *

logger = logging.getLogger(__name__)


@click.command()
@click.option("--output-dir", type=str, help="Path to save checkpoints", required=True)
@click.option(
    "--source-max-token-length",
    type=int,
    default=256,
    help="Maximum sequence length - source",
)
@click.option(
    "--target-max-token-length",
    type=int,
    default=256,
    help="Maximum sequence length - target",
)
@click.option(
    "--batch-size", type=int, default=12, help="Batch size to use for training"
)
@click.option("--max-epochs", type=int, default=1, help="Number of epochs to train for")
@click.option(
    "--learning-rate",
    type=float,
    default=3e-4,
    help="Learning rate to use for training",
)
@click.option(
    "--weight-decay", type=float, default=0.1, help="Weight decay to use for training"
)
@click.option(
    "--adam-epsilon", type=float, default=9e-7, help="Adam epsilon to use for training"
)
@click.option("--warmup-steps", type=int, default=0, help="Warm up steps for training")
@click.option(
    "--gradient-accumulation-steps",
    type=int,
    default=16,
    help="Gradient accumulation steps",
)
@click.option("--n-gpu", type=int, default=-1, help="Number of GPUs per device")
@click.option("--num-nodes", type=int, default=4, help="Number of machines")
@click.option(
    "--early-stopping-patience-epochs",
    type=int,
    default=0,
    help="Early stopping patience epochs",
)
@click.option("--precision", type=str, default="bf16", help="Precision")
@click.option("--logger", type=str, default="default", help="Logger")
@click.option(
    "--dataloader-num-workers",
    type=int,
    default=2,
    help="Number of workers for DataLoader",
)
@click.option("--opt-level", type=str, default="01", help="Opt level")
@click.option("--max-grad-norm", type=float, default=0.5, help="Max grad norm")
@click.option("--seed", type=int, default=42, help="Seed")
@click.option(
    "--early-stop-callback", type=bool, default=False, help="Early stop callback"
)
@click.option(
    "--save-only-last-epoch", type=bool, default=False, help="Save only last epoch"
)
@click.option("--fp-16", type=bool, default=False, help="Floating point 16 enabled")
@click.option("--use-gpu", is_flag=True, show_default=True, default=False, help="Use GPU")
def main(**kwargs):
    model = MistralFineTune(MODEL_NAME)
    model.train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
