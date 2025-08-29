import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.utils import str2bool


class BaseDatasetWrapper(Dataset):
    """Base wrapper for converting datasets to tensor format."""

    def __init__(self, device, data, targets, classes, transform=None, flatten=True):
        self.data = data
        self.targets = targets
        self.classes = classes
        self.transform = transform
        self.device = device
        self.flatten = flatten

    def __len__(self):
        return len(self.data)

    def _one_hot_encode(self, target):
        target_idx = target.item() if torch.is_tensor(target) else target
        return torch.eye(len(self.classes))[target_idx]

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__")


class MNISTDatasetWrapper(BaseDatasetWrapper):
    """MNIST-specific dataset wrapper."""

    def __getitem__(self, idx):
        img = self.data[idx]  # Tensor (28, 28)
        target = self.targets[idx]

        # Convert to PIL Image
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform:
            img = self.transform(img)  # Returns tensor (1, 28, 28)

        target = self._one_hot_encode(target)

        if self.flatten:
            img = img.view(28 * 28)

        return img, target


class CIFAR10DatasetWrapper(BaseDatasetWrapper):
    """CIFAR-10-specific dataset wrapper."""

    def __getitem__(self, idx):
        img = self.data[idx]  # Tensor (32, 32, 3) or (3, 32, 32)
        target = self.targets[idx]

        # Convert to PIL Image
        img = img.numpy() if torch.is_tensor(img) else img
        if img.shape[0] in [1, 3]:  # Channel-first to channel-last
            img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)  # Returns tensor (3, 32, 32)

        target = self._one_hot_encode(target)

        if self.flatten:
            img = img.view(3 * 32 * 32)

        return img, target


class BaseContinualDataloader:
    """Base class for continual learning dataloaders."""

    def __init__(self, config, dataset_name="MNIST"):
        self.config = config
        self.dataset_name = dataset_name
        self.num_tasks = 5
        self.batch_size = config.batch_size
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.classes_per_task = 2

        # Set flatten based on config and dataset
        if dataset_name == "MNIST":
            self.flatten = (
                True
                if config.get("flatten_imgs") == "default"
                else str2bool(config.get("flatten_imgs"))
            )
            self.in_channels = 1
            self.img_size = 28
            self.dataset_wrapper = MNISTDatasetWrapper
        else:  # CIFAR10
            self.flatten = (
                False
                if config.get("flatten_imgs") == "default"
                else str2bool(config.get("flatten_imgs"))
            )
            self.in_channels = 3
            self.img_size = 32
            self.dataset_wrapper = CIFAR10DatasetWrapper

        # Set input channels for config
        config.in_channels = self.in_channels

        self._setup_transforms()
        self._load_datasets()
        self._define_tasks()

    def _setup_transforms(self):
        """Setup transforms based on dataset."""
        if self.dataset_name == "MNIST":
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        else:  # CIFAR10
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                ]
            )

    def _load_datasets(self):
        """Load the appropriate dataset."""
        if self.dataset_name == "MNIST":
            self.train_dataset = datasets.MNIST(
                root="./data", train=True, transform=self.transform, download=True
            )
            self.test_dataset = datasets.MNIST(
                root="./data", train=False, transform=self.transform, download=True
            )
        else:  # CIFAR10
            self.train_dataset = datasets.CIFAR10(
                root="./data", train=True, transform=self.transform, download=True
            )
            self.test_dataset = datasets.CIFAR10(
                root="./data", train=False, transform=self.transform, download=True
            )

    def _define_tasks(self):
        """Define task splits - to be overridden by subclasses."""
        self.tasks = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def _filter_dataset(self, dataset, classes):
        """Filter dataset to only include specified classes."""
        indices = []
        for i, (_, target) in enumerate(dataset):
            if target in classes:
                indices.append(i)
        return Subset(dataset, indices)

    def _one_hot_encode(self, targets, num_classes):
        """Convert targets to one-hot encoding."""
        return torch.eye(num_classes)[targets].float()

    def _create_dataloader(self, tensor_dataset, shuffle=True):
        """Create a dataloader with consistent settings."""
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            generator = torch.Generator(device="cpu")

        return DataLoader(
            dataset=tensor_dataset,
            batch_size=self.batch_size,
            generator=generator.manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=shuffle,
        )

    def _process_data(self, dataset, classes, target_mapping_fn):
        """Process and extract data from filtered dataset."""
        data = torch.stack([dataset[i][0] for i in range(len(dataset))])
        targets = torch.tensor([target_mapping_fn(i) for i in range(len(dataset))])

        # Flatten if required
        if self.flatten:
            data = data.view(data.size(0), -1)

        return data, targets

    def get_dataloaders(self, task_id):
        """Get dataloaders for a specific task - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement get_dataloaders")

    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [self.get_dataloaders(task_id) for task_id in range(self.num_tasks)]


class DomainILDataloader(BaseContinualDataloader):
    """Domain Incremental Learning dataloader."""

    def get_dataloaders(self, task_id):
        classes = self.tasks[task_id]

        # Filter datasets
        train_dataset = self._filter_dataset(self.train_dataset, classes)
        test_dataset = self._filter_dataset(self.test_dataset, classes)

        # Process data with binary remapping (0 vs 1 for each task)
        train_data, train_targets = self._process_data(
            train_dataset,
            classes,
            lambda i: classes.index(
                self.train_dataset.targets[train_dataset.indices[i]].item()
            ),
        )

        test_data, test_targets = self._process_data(
            test_dataset,
            classes,
            lambda i: classes.index(
                self.test_dataset.targets[test_dataset.indices[i]].item()
            ),
        )

        # Create tensor datasets with binary one-hot encoding
        train_tensor_dataset = TensorDataset(
            train_data.float(),
            self._one_hot_encode(train_targets, 2),  # Always 2 classes for Domain IL
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, 2)
        )

        return (
            self._create_dataloader(train_tensor_dataset, shuffle=True),
            self._create_dataloader(test_tensor_dataset, shuffle=False),
        )


class TaskILDataloader(BaseContinualDataloader):
    """Task Incremental Learning dataloader."""

    def get_dataloaders(self, task_id):
        classes = self.tasks[task_id]

        # Filter datasets
        train_dataset = self._filter_dataset(self.train_dataset, classes)
        test_dataset = self._filter_dataset(self.test_dataset, classes)

        # Process data with binary remapping (same as Domain IL)
        train_data, train_targets = self._process_data(
            train_dataset,
            classes,
            lambda i: classes.index(
                self.train_dataset.targets[train_dataset.indices[i]].item()
            ),
        )

        test_data, test_targets = self._process_data(
            test_dataset,
            classes,
            lambda i: classes.index(
                self.test_dataset.targets[test_dataset.indices[i]].item()
            ),
        )

        # Create tensor datasets with binary one-hot encoding
        train_tensor_dataset = TensorDataset(
            train_data.float(),
            self._one_hot_encode(train_targets, 2),  # Always 2 classes per task
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, 2)
        )

        return (
            self._create_dataloader(train_tensor_dataset, shuffle=True),
            self._create_dataloader(test_tensor_dataset, shuffle=False),
        )


class ClassILDataloader(BaseContinualDataloader):
    """Class Incremental Learning dataloader."""

    def get_dataloaders(self, task_id):
        # Calculate all classes seen so far
        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.tasks[i])

        current_task_classes = self.tasks[task_id]
        num_classes_so_far = len(all_classes_so_far)

        # Filter datasets to current task's classes only
        train_dataset = self._filter_dataset(self.train_dataset, current_task_classes)
        test_dataset = self._filter_dataset(self.test_dataset, current_task_classes)

        # Process data keeping original labels (no remapping)
        train_data, train_targets = self._process_data(
            train_dataset,
            current_task_classes,
            lambda i: self.train_dataset.targets[train_dataset.indices[i]].item(),
        )

        test_data, test_targets = self._process_data(
            test_dataset,
            current_task_classes,
            lambda i: self.test_dataset.targets[test_dataset.indices[i]].item(),
        )

        # Create tensor datasets with growing one-hot encoding
        train_tensor_dataset = TensorDataset(
            train_data.float(), self._one_hot_encode(train_targets, num_classes_so_far)
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(), self._one_hot_encode(test_targets, num_classes_so_far)
        )

        return (
            self._create_dataloader(train_tensor_dataset, shuffle=True),
            self._create_dataloader(test_tensor_dataset, shuffle=False),
        )


# Factory functions to maintain compatibility with existing code
def DomainILMNIST(config):
    return DomainILDataloader(config, "MNIST")


def TaskILMNIST(config):
    return TaskILDataloader(config, "MNIST")


def ClassILMNIST(config):
    return ClassILDataloader(config, "MNIST")


def TaskILCIFAR10(config):
    return TaskILDataloader(config, "CIFAR10")


def ClassILCIFAR10(config):
    return ClassILDataloader(config, "CIFAR10")
