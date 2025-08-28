import torch
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from src.utils import str2bool

# Base dataset class (handles image conversion)
### CIFAR10 datasets can be found below. 

class BaseMNISTDataset(Dataset):
    def __init__(self, device, data, targets, classes, transform=None, setting = None, flatten = True):
        self.data = data  # Move data to specified device
        self.targets = targets
        self.classes = classes  # Classes present in this task
        self.transform = transform
        self.device = device 
        self.setting = setting
        self.flatten = flatten
                
    def __len__(self):
        return len(self.data)
    
    def _one_hot_encode(self, target):
        target_idx = target.item() if torch.is_tensor(target) else target
        return torch.eye(len(self.classes))[target_idx]
    
    def __getitem__(self, idx):
        # Fetch data on CPU
        img = self.data[idx]  # Tensor of shape (28, 28), on CPU
        target = self.targets[idx]  # Scalar tensor, on CPU
        # Convert to PIL Image (works on CPU)
        img = Image.fromarray(img.numpy(), mode='L')
        # Apply transform and move to device
        if self.transform:
            img = self.transform(img)  # Returns tensor (1, 28, 28)
        # One-hot encode and move to device
        target = self._one_hot_encode(target)
        if self.flatten:
            img = img.view(28 * 28)  # Flatten, still on device
        return img, target

# Task Incremental Learning class
class TaskILMNIST:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.batch_size = self.config.batch_size
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_per_task = 2
        self.flatten = True if self.config.get('flatten_imgs') == "default" else str2bool(self.config.get('flatten_imgs'))
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = self._load_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def _load_data(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        return (train_dataset.data, train_dataset.targets), (test_dataset.data, test_dataset.targets)
    
    def get_dataloaders(self, task_id):
        
        task_splits = []

        class_start = task_id * self.classes_per_task
        class_end = (task_id + 1) * self.classes_per_task
        task_classes = list(range(class_start, class_end))
        task_splits.append(task_classes)
        
        # Filter data for this task
        train_mask = torch.isin(self.train_targets, torch.tensor(task_classes))
        test_mask = torch.isin(self.test_targets, torch.tensor(task_classes))
        
        train_task_data = self.train_data[train_mask]
        train_task_targets = self.train_targets[train_mask]
        test_task_data = self.test_data[test_mask]
        test_task_targets = self.test_targets[test_mask]
        
        # Remap labels to [0, 1]
        label_map = {cls: idx for idx, cls in enumerate(task_classes)}
        train_task_targets = torch.tensor([label_map[t.item()] for t in train_task_targets])
        test_task_targets = torch.tensor([label_map[t.item()] for t in test_task_targets])
        
        # For one-hot encoding, classes should reflect the target range [0, 1]
        one_hot_classes = list(range(self.classes_per_task))  # [0, 1] for 2 classes

        # Create datasets
        train_dataset = BaseMNISTDataset(self.device, train_task_data, train_task_targets, one_hot_classes, self.transform, flatten=self.flatten)
        test_dataset = BaseMNISTDataset(self.device, test_task_data, test_task_targets, one_hot_classes, self.transform, flatten=self.flatten)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, 
                                  num_workers=self.config.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                 num_workers=self.config.num_workers)
        
        return train_loader, test_loader
    
    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [
            self.get_dataloaders(task_id) for task_id in range(self.num_tasks)
        ]

class DomainILMNIST:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.batch_size = self.config.batch_size
        self.flatten = True if self.config.get('flatten_imgs') == "default" else str2bool(self.config.get('flatten_imgs'))
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        
        # Load datasets
        self.train_dataset = datasets.MNIST(
            root="./data", train=True, transform=self.transform, download=True
        )
        self.test_dataset = datasets.MNIST(
            root="./data", train=False, transform=self.transform, download=True
        )
        
        # Define tasks as pairs of classes for even/odd classification
        self.tasks = [
            [0, 1],  # Task 0: 0 (even) vs 1 (odd)
            [2, 3],  # Task 1: 2 (even) vs 3 (odd)  
            [4, 5],  # Task 2: 4 (even) vs 5 (odd)
            [6, 7],  # Task 3: 6 (even) vs 7 (odd)
            [8, 9],  # Task 4: 8 (even) vs 9 (odd)
        ]
        
        # Set input channels for config
        config.in_channels = 1
        
    def _filter_dataset(self, dataset, classes):
        """Filter dataset to only include specified classes."""
        indices = []
        for i, (_, target) in enumerate(dataset):
            if target in classes:
                indices.append(i)
        return Subset(dataset, indices)
    
    def _one_hot_encode(self, targets):
        """Convert targets to one-hot encoding for binary classification."""
        return torch.eye(2)[targets].float()
        
    def get_dataloaders(self, task_id):
        """Get train/test dataloaders for a specific task with even/odd binary classification."""
        classes = self.tasks[task_id]
        
        # Filter datasets to only include the two classes for this task
        train_dataset = self._filter_dataset(self.train_dataset, classes)
        test_dataset = self._filter_dataset(self.test_dataset, classes)
        
        # Process training data
        train_data = torch.stack([train_dataset[i][0] for i in range(len(train_dataset))])
        train_targets = torch.tensor([
            classes.index(self.train_dataset.targets[train_dataset.indices[i]].item())
            for i in range(len(train_dataset))
        ])
        
        # Process test data
        test_data = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))])
        test_targets = torch.tensor([
            classes.index(self.test_dataset.targets[test_dataset.indices[i]].item())
            for i in range(len(test_dataset))
        ])
        
        # Flatten data if required
        if self.flatten:
            train_data = train_data.view(-1, 28 * 28)
            test_data = test_data.view(-1, 28 * 28)
        
        # Create tensor datasets with one-hot encoded targets (binary classification)
        train_tensor_dataset = TensorDataset(
            train_data.float(),
            self._one_hot_encode(train_targets)
        )
        test_tensor_dataset = TensorDataset(
            test_data.float(),
            self._one_hot_encode(test_targets)
        )
        
        # Create device-aware generator
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            # Fallback to CPU generator if device generator not supported
            generator = torch.Generator(device='cpu')
        
        # Create dataloaders
        train_loader = DataLoader(
            dataset=train_tensor_dataset,
            batch_size=self.batch_size,
            generator=generator.manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=True,
        )
        
        test_loader = DataLoader(
            dataset=test_tensor_dataset,
            batch_size=self.batch_size,
            generator=generator.manual_seed(self.config.seed),
            num_workers=self.config.num_workers,
            pin_memory=True,
            shuffle=False,
        )
        
        return train_loader, test_loader
    
    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [
            self.get_dataloaders(task_id) for task_id in range(self.num_tasks)
        ]

# Class Incremental Learning class
class ClassILMNIST:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.batch_size = self.config.batch_size
        self.classes_per_task = 2
        self.flatten = True if self.config.get('flatten_imgs') == "default" else str2bool(self.config.get('flatten_imgs'))
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = self._load_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def _load_data(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        return (train_dataset.data, train_dataset.targets), (test_dataset.data, test_dataset.targets)
    
    def get_dataloaders(self, task_id):
        
        # Define task splits (e.g., 0-1, 2-3, 4-5, 6-7, 8-9)
        class_start = task_id * self.classes_per_task
        class_end = (task_id + 1) * self.classes_per_task

        task_classes = list(range(class_start, class_end))
        # Filter data for this task
        train_mask = torch.isin(self.train_targets, torch.tensor(task_classes))
        test_mask = torch.isin(self.test_targets, torch.tensor(task_classes))
        
        train_task_data = self.train_data[train_mask]
        train_task_targets = self.train_targets[train_mask]
        test_task_data = self.test_data[test_mask]
        test_task_targets = self.test_targets[test_mask]
        
        # Define all possible classes (0-9) for one-hot encoding consistency
        all_classes = list(range(10))

        # No label remapping (keep original labels)
        train_dataset = BaseMNISTDataset(self.device, train_task_data, train_task_targets, all_classes, self.transform, setting = "class", flatten = self.flatten)
        test_dataset = BaseMNISTDataset(self.device, test_task_data, test_task_targets, all_classes, self.transform, setting = "class", flatten = self.flatten)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, 
                                  num_workers=self.config.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                 num_workers=self.config.num_workers)
        
        return train_loader, test_loader
        
    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [
            self.get_dataloaders(task_id) for task_id in range(self.num_tasks)
        ]
        


class BaseCIFAR10Dataset(Dataset):
    def __init__(self, device, data, targets, classes, transform=None, setting=None, flatten = False):
        self.data = data  # Move data to specified device
        self.targets = targets
        self.classes = classes  # Classes present in this task
        self.transform = transform
        self.device = device 
        self.setting = setting
        self.flatten = flatten
        
    def __len__(self):
        return len(self.data)
    
    def _one_hot_encode(self, target):
        target_idx = target.item() if torch.is_tensor(target) else target
        return torch.eye(len(self.classes))[target_idx]
    
    def __getitem__(self, idx):
        # Fetch data on CPU
        img = self.data[idx]  # Tensor of shape (3, 32, 32) or (32, 32, 3) depending on format, on CPU
        target = self.targets[idx]  # Scalar tensor, on CPU
        # Convert to PIL Image (works on CPU), ensure channel-last format (H, W, C)
        img = img.numpy() if torch.is_tensor(img) else img
        if img.shape[0] in [1, 3]:  # If channel-first, transpose to (32, 32, 3)
            img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)  # CIFAR-10 uses RGB
        # Apply transform and move to device
        if self.transform:
            img = self.transform(img)  # Returns tensor (3, 32, 32)
        # One-hot encode and move to device
        target = self._one_hot_encode(target)
        if self.flatten:
            img = img.view(3 * 32 * 32)  # Flatten to (3072,), still on device
        return img, target

# Task Incremental Learning class
class TaskILCIFAR10:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.flatten = False if config.get('flatten_imgs') == "default" else str2bool(self.config.get('flatten_imgs'))
        self.batch_size = self.config.batch_size
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_per_task = 2
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = self._load_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            ])
        
    def _load_data(self):
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        # CIFAR-10 data is numpy arrays of shape (N, 32, 32, 3), convert to tensors
        train_data = torch.tensor(train_dataset.data, dtype=torch.uint8)
        train_targets = torch.tensor(train_dataset.targets, dtype=torch.long)
        test_data = torch.tensor(test_dataset.data, dtype=torch.uint8)
        test_targets = torch.tensor(test_dataset.targets, dtype=torch.long)
        return (train_data, train_targets), (test_data, test_targets)
    
    def get_dataloaders(self, task_id):
        task_splits = []

        class_start = task_id * self.classes_per_task
        class_end = (task_id + 1) * self.classes_per_task
        task_classes = list(range(class_start, class_end))
        task_splits.append(task_classes)
        
        # Filter data for this task
        train_mask = torch.isin(self.train_targets, torch.tensor(task_classes))
        test_mask = torch.isin(self.test_targets, torch.tensor(task_classes))
        
        train_task_data = self.train_data[train_mask]
        train_task_targets = self.train_targets[train_mask]
        test_task_data = self.test_data[test_mask]
        test_task_targets = self.test_targets[test_mask]
        
        # Remap labels to [0, 1]
        label_map = {cls: idx for idx, cls in enumerate(task_classes)}
        train_task_targets = torch.tensor([label_map[t.item()] for t in train_task_targets])
        test_task_targets = torch.tensor([label_map[t.item()] for t in test_task_targets])
        
        # For one-hot encoding, classes should reflect the target range [0, 1]
        one_hot_classes = list(range(self.classes_per_task))  # [0, 1] for 2 classes

        # Create datasets
        train_dataset = BaseCIFAR10Dataset(self.device, train_task_data, train_task_targets, one_hot_classes, self.transform, self.flatten)
        test_dataset = BaseCIFAR10Dataset(self.device, test_task_data, test_task_targets, one_hot_classes, self.transform, self.flatten)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, 
                                  num_workers=self.config.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                 num_workers=self.config.num_workers)
        
        return train_loader, test_loader
    
    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [
            self.get_dataloaders(task_id) for task_id in range(self.num_tasks)
        ]
        
class ClassILCIFAR10:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.flatten = False if self.config.get('flatten_imgs') == "default" else str2bool(self.config.get('flatten_imgs'))
        self.batch_size = self.config.batch_size
        self.classes_per_task = 2
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = self._load_data()
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Scales [0, 255] to [0, 1] if not already normalized
            # No normalization applied
        ])
        
    def _load_data(self):
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)
        # CIFAR-10 data is numpy arrays of shape (N, 32, 32, 3), convert to tensors
        train_data = torch.tensor(train_dataset.data, dtype=torch.uint8)
        train_targets = torch.tensor(train_dataset.targets, dtype=torch.long)
        test_data = torch.tensor(test_dataset.data, dtype=torch.uint8)
        test_targets = torch.tensor(test_dataset.targets, dtype=torch.long)
        return (train_data, train_targets), (test_data, test_targets)
    
    def get_dataloaders(self, task_id):
        # Define task splits (e.g., 0-1, 2-3, 4-5, 6-7, 8-9)
        class_start = task_id * self.classes_per_task
        class_end = (task_id + 1) * self.classes_per_task

        task_classes = list(range(class_start, class_end))
        # Filter data for this task
        train_mask = torch.isin(self.train_targets, torch.tensor(task_classes))
        test_mask = torch.isin(self.test_targets, torch.tensor(task_classes))
        
        train_task_data = self.train_data[train_mask]
        train_task_targets = self.train_targets[train_mask]
        test_task_data = self.test_data[test_mask]
        test_task_targets = self.test_targets[test_mask]
        
        # Define all possible classes (0-9) for one-hot encoding consistency
        all_classes = list(range(10))

        # No label remapping (keep original labels)
        train_dataset = BaseCIFAR10Dataset(self.device, train_task_data, train_task_targets, all_classes, self.transform, setting="class", flatten = self.flatten)
        test_dataset = BaseCIFAR10Dataset(self.device, test_task_data, test_task_targets, all_classes, self.transform, setting="class", flatten = self.flatten)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False, 
                                  num_workers=self.config.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                 num_workers=self.config.num_workers)
        
        return train_loader, test_loader
        
    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [
            self.get_dataloaders(task_id) for task_id in range(self.num_tasks)
        ]