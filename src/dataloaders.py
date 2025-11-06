import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.utils import str2bool


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
            shuffle=True,
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=self.config.batch_size,
            generator=torch.Generator(device=self.config.device).manual_seed(
                self.config.seed
            ),
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        return train_loader, test_loader

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
        self.num_tasks = config.num_tasks
        self.classes_per_task = config.classes_per_task
        self.batch_size = config.batch_size
        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

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
        self._precompute_task_indices()

    def _setup_transforms(self):
        """Setup transforms based on dataset."""
        if self.dataset_name == "MNIST":
            self.transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )
        else:  # CIFAR10
            self.transform = transforms.Compose([transforms.ToTensor()])

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

    def _precompute_task_indices(self):
        """Pre-compute indices for each task to avoid repeated filtering."""
        self.task_train_indices = {}
        self.task_test_indices = {}
        
        for task_id in range(self.num_tasks):
            task_classes = self.tasks[task_id]
            
            # Pre-filter train indices
            self.task_train_indices[task_id] = [
                i for i, target in enumerate(self.train_dataset.targets)
                if target in task_classes
            ]
            
            # Pre-filter test indices  
            self.task_test_indices[task_id] = [
                i for i, target in enumerate(self.test_dataset.targets)
                if target in task_classes
            ]

    def _get_task_subset(self, is_train, task_id):
        """Get dataset subset for a specific task using pre-computed indices."""
        indices = self.task_train_indices[task_id] if is_train else self.task_test_indices[task_id]
        dataset = self.train_dataset if is_train else self.test_dataset
        return Subset(dataset, indices)

    def _get_cumulative_test_subset(self, up_to_task_id):
        """Get test subset including all classes up to specified task (for Class IL)."""
        all_indices = []
        for task_id in range(up_to_task_id + 1):
            all_indices.extend(self.task_test_indices[task_id])
        return Subset(self.test_dataset, all_indices)

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

        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        test_dataset = self._get_task_subset(is_train=False, task_id=task_id)

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
        train_dataset = self._get_task_subset(is_train=True, task_id=task_id) 
        test_dataset = self._get_task_subset(is_train=False, task_id=task_id)

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
        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.tasks[i])

        # Training: only current task
        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        # Testing: all seen classes so far
        test_dataset = self._get_cumulative_test_subset(up_to_task_id=task_id)
        num_classes_so_far = (task_id + 1) * self.classes_per_task

        train_data, train_targets = self._process_data(
            train_dataset, self.tasks[task_id],
            lambda i: self.train_dataset.targets[train_dataset.indices[i]].item()
        )

        test_data, test_targets = self._process_data(
            test_dataset, all_classes_so_far,
            lambda i: self.test_dataset.targets[test_dataset.indices[i]].item()
        )

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


class ClassIL5TaskDataloader(BaseContinualDataloader):
    """5-task Class Incremental Learning dataloader (2 classes per task)."""
    
    def __init__(self, config, dataset_name="MNIST"):
        super().__init__(config, dataset_name)
        self.num_tasks = 5
        self.classes_per_task = 2

    def get_dataloaders(self, task_id):
        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.tasks[i])
        
        # Training: only current task
        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        # Testing: all seen classes so far
        test_dataset = self._get_cumulative_test_subset(up_to_task_id=task_id)
        num_classes_so_far = (task_id + 1) * self.classes_per_task

        train_data, train_targets = self._process_data(
            train_dataset, self.tasks[task_id],
            lambda i: self.train_dataset.targets[train_dataset.indices[i]].item()
        )

        test_data, test_targets = self._process_data(
            test_dataset, all_classes_so_far,
            lambda i: self.test_dataset.targets[test_dataset.indices[i]].item()
        )

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


class ClassIL2TaskDataloader(BaseContinualDataloader):
    """2-task Class Incremental Learning dataloader (5 classes per task)."""
    
    def __init__(self, config, dataset_name="MNIST"):
        # Set task configuration before calling super()
        self.num_tasks = 2
        self.classes_per_task = 5
        super().__init__(config, dataset_name)
        
    def _define_tasks(self):
        """Override to define 2 tasks with 5 classes each."""
        self.tasks = [
            [0, 1, 2, 3, 4],  # Task 0: first 5 digits
            [5, 6, 7, 8, 9]   # Task 1: second 5 digits
        ]

    def get_dataloaders(self, task_id):
        all_classes_so_far = []
        for i in range(task_id + 1):
            all_classes_so_far.extend(self.tasks[i])
        
        # Training: only current task
        train_dataset = self._get_task_subset(is_train=True, task_id=task_id)
        # Testing: all seen classes so far
        test_dataset = self._get_cumulative_test_subset(up_to_task_id=task_id)
        num_classes_so_far = (task_id + 1) * self.classes_per_task

        train_data, train_targets = self._process_data(
            train_dataset, self.tasks[task_id],
            lambda i: self.train_dataset.targets[train_dataset.indices[i]].item()
        )

        test_data, test_targets = self._process_data(
            test_dataset, all_classes_so_far,
            lambda i: self.test_dataset.targets[test_dataset.indices[i]].item()
        )

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


def ClassILMNIST5Task(config):
    return ClassIL5TaskDataloader(config, "MNIST")

def ClassILMNIST2Task(config):
    return ClassIL2TaskDataloader(config, "MNIST")

def TaskILCIFAR10(config):
    return TaskILDataloader(config, "CIFAR10")

def ClassILCIFAR105Task(config):
    return ClassIL5TaskDataloader(config, "CIFAR10")

def ClassILCIFAR102Task(config):
    return ClassIL2TaskDataloader(config, "CIFAR10")