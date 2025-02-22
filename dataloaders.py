import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Base dataset class (handles image conversion)

class BaseMNISTDataset(Dataset):
    def __init__(self, data, targets, classes, transform=None):
        self.data = data  # Tensor of shape (N, 28, 28)
        self.targets = targets  # Targets (remapped or original)
        self.classes = classes  # Classes present in this task
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    # def _one_hot_encode(self, targets):
    #     return torch.eye(len(self.classes))[targets]

    def _one_hot_encode(self, target):
        """Convert a scalar target to one-hot encoding based on self.classes."""
        # Map target to index in self.classes
        # target_idx = self.classes.index(target.item() if torch.is_tensor(target) else target)
        # return torch.eye(len(self.classes))[target_idx]
        target_idx = target.item() if torch.is_tensor(target) else target
        return torch.eye(len(self.classes))[target_idx]
    
    def __getitem__(self, idx):
        img = self.data[idx]  # Tensor of shape (28, 28)
        img = Image.fromarray(img.numpy(), mode='L')  # Convert to PIL Image (grayscale)
        target = self._one_hot_encode(self.targets[idx])
        if self.transform:
            img = self.transform(img)
        img = img.view(28 * 28)
        return img, target

# Task Incremental Learning class
class TaskILMNIST:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.batch_size = self.config.batch_size
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.classes_per_task = 2
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def _load_data(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        return (train_dataset.data, train_dataset.targets), (test_dataset.data, test_dataset.targets)
    
    def get_dataloaders(self, task_id):
        (train_data, train_targets), (test_data, test_targets) = self._load_data()
        
        task_splits = []

        class_start = task_id * self.classes_per_task
        class_end = (task_id + 1) * self.classes_per_task
        task_classes = list(range(class_start, class_end))
        task_splits.append(task_classes)
        
        # Filter data for this task
        train_mask = torch.isin(train_targets, torch.tensor(task_classes))
        test_mask = torch.isin(test_targets, torch.tensor(task_classes))
        
        train_task_data = train_data[train_mask]
        train_task_targets = train_targets[train_mask]
        test_task_data = test_data[test_mask]
        test_task_targets = test_targets[test_mask]
        
        # Remap labels to [0, 1]
        label_map = {cls: idx for idx, cls in enumerate(task_classes)}
        train_task_targets = torch.tensor([label_map[t.item()] for t in train_task_targets])
        test_task_targets = torch.tensor([label_map[t.item()] for t in test_task_targets])
        
        # For one-hot encoding, classes should reflect the target range [0, 1]
        one_hot_classes = list(range(self.classes_per_task))  # [0, 1] for 2 classes

        # Create datasets
        train_dataset = BaseMNISTDataset(train_task_data, train_task_targets, one_hot_classes, self.transform)
        test_dataset = BaseMNISTDataset(test_task_data, test_task_targets, one_hot_classes, self.transform)
        
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            # Fallback to CPU generator if device generator not supported
            generator = torch.Generator(device='cpu')

        # Create dataloaders
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

# Class Incremental Learning class
class DomainILMNIST:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.batch_size = self.config.batch_size
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        
    def _load_data(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        return (train_dataset.data, train_dataset.targets), (test_dataset.data, test_dataset.targets)
    
    def get_dataloaders(self, task_id):
        (train_data, train_targets), (test_data, test_targets) = self._load_data()
        
        # All classes (0-9) in each task
        task_splits = [list(range(10)) for _ in range(self.num_tasks)]
        task_classes = task_splits[task_id]

        # Filter data (all classes, but with domain shift)
        train_mask = torch.isin(train_targets, torch.tensor(task_classes))
        test_mask = torch.isin(test_targets, torch.tensor(task_classes))
        
        train_task_data = train_data[train_mask]
        train_task_targets = train_targets[train_mask]
        test_task_data = test_data[test_mask]
        test_task_targets = test_targets[test_mask]
        
        # Domain-specific transform (rotation as domain shift)
        transform = transforms.Compose([
            transforms.RandomRotation(degrees=(task_id * 10, task_id * 10 + 10)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Create datasets (no label remapping)
        train_dataset = BaseMNISTDataset(train_task_data, train_task_targets, task_classes, transform)
        test_dataset = BaseMNISTDataset(test_task_data, test_task_targets, task_classes, transform)
        
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            # Fallback to CPU generator if device generator not supported
            generator = torch.Generator(device='cpu')

        # Create dataloaders
        # train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, 
        #                           num_workers=self.config.num_workers, generator=generator.manual_seed(self.config.seed))
        # test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, 
        #                          num_workers=self.config.num_workers, generator=generator.manual_seed(self.config.seed))
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

# Domain Incremental Learning class
class ClassILMNIST:
    def __init__(self, config):
        self.num_tasks = 5
        self.config = config
        self.batch_size = self.config.batch_size
        self.classes_per_task = 2
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def _load_data(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)
        return (train_dataset.data, train_dataset.targets), (test_dataset.data, test_dataset.targets)
    
    def get_dataloaders(self, task_id):
        (train_data, train_targets), (test_data, test_targets) = self._load_data()
        
        # Define task splits (e.g., 0-1, 2-3, 4-5, 6-7, 8-9)
        class_start = task_id * self.classes_per_task
        class_end = (task_id + 1) * self.classes_per_task

        task_classes = list(range(class_start, class_end))
        # Filter data for this task
        train_mask = torch.isin(train_targets, torch.tensor(task_classes))
        test_mask = torch.isin(test_targets, torch.tensor(task_classes))
        
        train_task_data = train_data[train_mask]
        train_task_targets = train_targets[train_mask]
        test_task_data = test_data[test_mask]
        test_task_targets = test_targets[test_mask]

        # Remap labels to [0, 1] based on even/odd parity
        # Even = 0, Odd = 1
        train_task_targets = torch.tensor([0 if t.item() % 2 == 0 else 1 for t in train_task_targets])
        test_task_targets = torch.tensor([0 if t.item() % 2 == 0 else 1 for t in test_task_targets])
        
        # Use [0, 1] as classes for even/odd classification
        even_odd_classes = [0, 1]

        # No label remapping (keep original labels)
        train_dataset = BaseMNISTDataset(train_task_data, train_task_targets, even_odd_classes, self.transform)
        test_dataset = BaseMNISTDataset(test_task_data, test_task_targets, even_odd_classes, self.transform)
        
        try:
            generator = torch.Generator(device=self.device)
        except RuntimeError:
            # Fallback to CPU generator if device generator not supported
            generator = torch.Generator(device='cpu')

        # Create dataloaders
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