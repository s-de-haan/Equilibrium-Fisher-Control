import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class BaseCIFAR10Dataset(Dataset):
    def __init__(self, device, data, targets, classes, transform=None, setting=None):
        self.data = data  # Move data to specified device
        self.targets = targets
        self.classes = classes  # Classes present in this task
        self.transform = transform
        self.device = device 
        self.setting = setting
        
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
        # img = img.view(3 * 32 * 32)  # Flatten to (3072,), still on device
        return img, target

# Task Incremental Learning class
class TaskILCIFAR10:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
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
        train_dataset = BaseCIFAR10Dataset(self.device, train_task_data, train_task_targets, one_hot_classes, self.transform)
        test_dataset = BaseCIFAR10Dataset(self.device, test_task_data, test_task_targets, one_hot_classes, self.transform)

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
        train_dataset = BaseCIFAR10Dataset(self.device, train_task_data, train_task_targets, all_classes, self.transform, setting="class")
        test_dataset = BaseCIFAR10Dataset(self.device, test_task_data, test_task_targets, all_classes, self.transform, setting="class")

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
        
class DomainILCIFAR10:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.batch_size = self.config.batch_size
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        (self.train_data, self.train_targets), (self.test_data, self.test_targets) = self._load_data()
        
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
        # All classes (0-9) in each task
        task_splits = [list(range(10)) for _ in range(self.num_tasks)]
        task_classes = task_splits[task_id]

        # Filter data (all classes, but with domain shift)
        train_mask = torch.isin(self.train_targets, torch.tensor(task_classes))
        test_mask = torch.isin(self.test_targets, torch.tensor(task_classes))
        
        train_task_data = self.train_data[train_mask]
        train_task_targets = self.train_targets[train_mask]
        test_task_data = self.test_data[test_mask]
        test_task_targets = self.test_targets[test_mask]
        
        # Domain-specific transform (rotation as domain shift)
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(task_id * 10, task_id * 10 + 10)),
            transforms.ToTensor(),  # Scales [0, 255] to [0, 1] if not already normalized
            # No normalization applied
        ])
        
        # Create datasets (no label remapping)
        train_dataset = BaseCIFAR10Dataset(self.device, train_task_data, train_task_targets, task_classes, transform)
        test_dataset = BaseCIFAR10Dataset(self.device, test_task_data, test_task_targets, task_classes, transform)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
                                  num_workers=self.config.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
                                 num_workers=self.config.num_workers)
        return train_loader, test_loader
    
    def get_all_tasks_dataloaders(self):
        """Get dataloaders for all tasks."""
        return [
            self.get_dataloaders(task_id) for task_id in range(self.num_tasks)
        ]