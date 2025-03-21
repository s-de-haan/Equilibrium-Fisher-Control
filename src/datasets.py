import torchvision.datasets as datasets
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Subset


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
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
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

    def get_task_dataloaders(self, task_id):
        """Get train/test dataloaders for a specific task."""
        classes = self.tasks[task_id]

        # Filter datasets
        train_dataset = self._filter_dataset(self.train_dataset, classes)
        test_dataset = self._filter_dataset(self.test_dataset, classes)

        # Process data
        train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_targets = torch.tensor([
            classes.index(self.train_dataset.targets[train_dataset.indices[i]].item())
            for i in range(len(train_dataset))
        ])

        test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
        test_targets = torch.tensor([
            classes.index(self.test_dataset.targets[test_dataset.indices[i]].item())
            for i in range(len(test_dataset))
        ])

        # Reshape data
        train_data = train_data.view(-1, 28 * 28)
        test_data = test_data.view(-1, 28 * 28)

        # Create datasets
        train_dataset = TensorDataset(
            train_data.float(),
            self._one_hot_encode(train_targets)
        )
        test_dataset = TensorDataset(
            test_data.float(),
            self._one_hot_encode(test_targets)
        )

        # Create dataloaders with device-aware seed
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            # Fallback to CPU generator if device generator not supported
            generator = torch.Generator(device='cpu')
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            generator=generator.manual_seed(self.config.seed),
            # num_workers=self.config.num_workers,
            # pin_memory=True,
            shuffle=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            generator=generator.manual_seed(self.config.seed),
            # num_workers=self.config.num_workers,
            # pin_memory=True,
            shuffle=False,
        )

        return train_loader, test_loader
        
    def _one_hot_encode(self, targets):
        """Convert target classes to one-hot encoded format."""
        n_classes = 2  # Binary classification for each task
        return torch.eye(n_classes)[targets]

    def _filter_dataset(self, dataset, classes):
        """Efficiently filter dataset for specific classes."""
        class_indices = np.isin(dataset.targets, classes)
        filtered_dataset = Subset(dataset, np.where(class_indices)[0])
        return filtered_dataset

    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [
            self.get_task_dataloaders(task_id)
            for task_id in range(len(self.tasks))
        ]