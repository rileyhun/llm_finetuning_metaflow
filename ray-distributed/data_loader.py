import os
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import MNIST
import pytorch_lightning as pl
from filelock import FileLock


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=100):
        super().__init__()
        self.data_dir = os.getcwd()
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        with FileLock(f"{self.data_dir}.lock"):
            mnist = MNIST(
                self.data_dir, train=True, download=True, transform=self.transform
            )

            # split data into train and val sets
            self.mnist_train, self.mnist_val = random_split(mnist, [55000, 5000])

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        with FileLock(f"{self.data_dir}.lock"):
            self.mnist_test = MNIST(
                self.data_dir, train=False, download=True, transform=self.transform
            )
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)