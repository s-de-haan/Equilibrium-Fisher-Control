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
        config.in_channels = 1

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
            transforms.Normalize((0.1307,), (0.3081,)),
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

        config.in_channels = 1

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
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            generator=generator.manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            pin_memory=True,
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

import torchvision.datasets as datasets
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset, Subset

class ConvMNISTWrapper:
    """Wrapper for MNIST and SplitMNIST to provide 4D input for convolutional networks."""
    
    def __init__(self, base_class, config):
        self.base = base_class(config)  # Instantiate the base class (MNIST or SplitMNIST)
        self.config = config
        self.device = config.device
        self.height = 28  # MNIST-specific
        self.width = 28
        self.in_channels = 1  # Grayscale
        
        # Ensure config reflects convolutional input
        config.in_channels = self.in_channels
        config.input_height = self.height
        config.input_width = self.width

    def _one_hot_encode(self, targets, n_classes):
        """Helper for one-hot encoding with configurable number of classes."""
        return torch.eye(n_classes)[targets]

    def get_dataloaders(self):
        """Override for MNIST: Return dataloaders with 4D input."""
        if not hasattr(self.base, 'get_dataloaders'):
            raise NotImplementedError("Base class must implement get_dataloaders or get_task_dataloaders")
        
        # Load data in original 4D format
        train_data = torch.stack(
            [self.base.train_dataset[i][0] for i in range(len(self.base.train_dataset))]
        )  # [N, 1, 28, 28]
        train_targets = torch.tensor(
            [self.base.train_dataset[i][1] for i in range(len(self.base.train_dataset))]
        )
        train_targets = self._one_hot_encode(train_targets, n_classes=10)

        test_data = torch.stack(
            [self.base.test_dataset[i][0] for i in range(len(self.base.test_dataset))]
        )  # [N, 1, 28, 28]
        test_targets = torch.tensor(
            [self.base.test_dataset[i][1] for i in range(len(self.base.test_dataset))]
        )
        test_targets = self._one_hot_encode(test_targets, n_classes=10)

        # Create datasets without flattening
        train_dataset = TensorDataset(train_data, train_targets)
        test_dataset = TensorDataset(test_data, test_targets)

        # Dataloaders
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
            shuffle=False,
        )

        return train_loader, test_loader

    def get_task_dataloaders(self, task_id):
        """Override for SplitMNIST: Return task-specific dataloaders with 4D input."""
        if not hasattr(self.base, 'get_task_dataloaders'):
            raise NotImplementedError("Base class must implement get_task_dataloaders")

        classes = self.base.tasks[task_id]

        # Filter datasets
        train_dataset = self.base._filter_dataset(self.base.train_dataset, classes)
        test_dataset = self.base._filter_dataset(self.base.test_dataset, classes)

        # Load data in 4D format
        train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_targets = torch.tensor([
            classes.index(self.base.train_dataset.targets[train_dataset.indices[i]].item())
            for i in range(len(train_dataset))
        ])
        train_targets = self._one_hot_encode(train_targets, n_classes=2)  # Binary for SplitMNIST

        test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
        test_targets = torch.tensor([
            classes.index(self.base.test_dataset.targets[test_dataset.indices[i]].item())
            for i in range(len(test_dataset))
        ])
        test_targets = self._one_hot_encode(test_targets, n_classes=2)

        # Create datasets
        train_dataset = TensorDataset(train_data.float(), train_targets)
        test_dataset = TensorDataset(test_data.float(), test_targets)

        # Device-aware generator
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            generator = torch.Generator(device='cpu')

        # Dataloaders
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            generator=generator.manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            generator=generator.manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=False,
        )

        return train_loader, test_loader

    def get_all_tasks_dataloaders(self):
        """Override for SplitMNIST: Return dataloaders for all tasks."""
        if not hasattr(self.base, 'get_all_tasks_dataloaders'):
            raise NotImplementedError("Base class must implement get_all_tasks_dataloaders")
        return [self.get_task_dataloaders(task_id) for task_id in range(len(self.base.tasks))]


class ConvMNIST(ConvMNISTWrapper):
    def __init__(self, config):
        super().__init__(MNIST, config)

class ConvSplitMNIST(ConvMNISTWrapper):
    def __init__(self, config):
        super().__init__(SplitMNIST, config)