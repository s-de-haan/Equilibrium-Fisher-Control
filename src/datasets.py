import torchvision.datasets as datasets
import numpy as np
import torch

from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms, datasets


class MNIST:
    def __init__(self, config):
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.train_dataset = datasets.MNIST(
            root="data", train=True, transform=transform, download=True
        )
        self.test_dataset = datasets.MNIST(
            root="data", train=False, transform=transform, download=True
        )

        self.config = config

    def _one_hot_encode(self, targets):
        n_classes = 10
        return torch.eye(n_classes)[targets]

    def get_dataloaders(self):
        train_data = torch.stack(
            [self.train_dataset[i][0] for i in range(len(self.train_dataset))]
        ).view(-1, 28 * 28)
        train_targets = self._one_hot_encode(
            torch.tensor(
                [self.train_dataset[i][1] for i in range(len(self.train_dataset))]
            )
        )

        test_data = torch.stack(
            [self.test_dataset[i][0] for i in range(len(self.test_dataset))]
        ).view(-1, 28 * 28)
        test_targets = self._one_hot_encode(
            torch.tensor(
                [self.test_dataset[i][1] for i in range(len(self.test_dataset))]
            )
        )

        train_dataset = TensorDataset(train_data, train_targets)
        test_dataset = TensorDataset(test_data, test_targets)

        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.config.device).manual_seed(
                self.config.seed
            ),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.config.device).manual_seed(
                self.config.seed
            ),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=False,
        )

        return train_loader, test_loader


class SplitMNIST:
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

        self.tasks = [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
            [8, 9],
        ]
        self.config = config

    def _one_hot_encode(self, targets):
        n_classes = 2
        return torch.eye(n_classes)[targets]

    def get_task_data(self, dataset, classes):
        """Filter dataset by specific classes."""
        indices = [i for i, target in enumerate(dataset.targets) if target in classes]
        data = dataset.data[indices].view(-1, 28 * 28)
        targets = dataset.targets[indices]
        return data, targets

    def get_task_dataloaders(self, task_id):
        """Get train/test dataloaders for a specific task."""
        classes = self.tasks[task_id]

        train_data, train_targets = self.get_task_data(self.train_dataset, classes)
        train_targets = torch.tensor([classes.index(t.item()) for t in train_targets])

        test_data, test_targets = self.get_task_data(self.test_dataset, classes)
        test_targets = torch.tensor([classes.index(t.item()) for t in test_targets])

        train_loader = DataLoader(
            TensorDataset(train_data.float(), self._one_hot_encode(train_targets)),
            batch_size=self.config["batch_size"],
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(test_data.float(), self._one_hot_encode(test_targets)),
            batch_size=self.config["batch_size"],
            shuffle=False,
        )

        return train_loader, test_loader

    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [
            self.get_task_dataloaders(task_id) for task_id in range(len(self.tasks))
        ]
