import torchvision.datasets as datasets
import numpy as np
import torch

from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

class MNIST:
    def __init__(self, config):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        self.train_dataset = datasets.MNIST(
            root="data", train=True, transform=transform, download=True
        )
        self.test_dataset = datasets.MNIST(
            root="data", train=False, transform=transform, download=True
        )

        self.train_dataset.data = self.train_dataset.data.float().view(-1, 28*28)
        self.test_dataset.data = self.test_dataset.data.float().view(-1, 28*28)

        self.train_dataset.targets = self._one_hot_encode(self.train_dataset.targets)
        self.test_dataset.targets = self._one_hot_encode(self.test_dataset.targets)

        self.config = config

    def _one_hot_encode(self, targets):
        n_classes = 10
        return torch.eye(n_classes)[targets]

    def get_dataloaders(self, batch_size):
        train_data, train_targets = self.train_dataset.data, self.train_dataset.targets
        test_data, test_targets = self.test_dataset.data, self.test_dataset.targets

        train_dataset = TensorDataset(train_data, train_targets)
        test_dataset = TensorDataset(test_data, test_targets)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.config.device).manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.config.device).manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        return train_loader, test_loader


class SplitMNIST:
    # TODO
    pass
