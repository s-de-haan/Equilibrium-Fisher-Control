from networks.network_interface import Network
from networks.layers import BP_layer
from networks.activation_function import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

class TaskIncrementalBP_layer(BP_layer):
    def __init__(self, in_features, out_features, activation_fn, name="TIL_BP_layer") -> None:
        super().__init__(in_features, out_features, activation_fn, name)
        self.task_id = None
        self.frozen = False
        
    def set_task(self, task_id):
        self.task_id = task_id
        
    def freeze(self):
        self.frozen = True
        # We don't actually detach parameters, but will handle this in optimizer
        
    def unfreeze(self):
        self.frozen = False
    
    def is_frozen(self):
        return self.frozen


class TaskIncrementalBP_network(Network):
    def __init__(self, config, num_tasks=5, task_output_size=2, name="TIL_BP_network") -> None:
        self.num_tasks = num_tasks
        self.task_output_size = task_output_size
        self.current_task = 0
        self.trained_tasks = set()
        
        # Modify the layers configuration to work with task-specific outputs
        # We'll create a shared feature extractor and separate output heads
        self.orig_layers = config.layers.copy()
        
        # Set the output size to be task_output_size (e.g., 2 for Split MNIST)
        # Last shared layer size
        self.feature_size = self.orig_layers[-2]
        
        super().__init__(TaskIncrementalBP_layer, ReLU, Linear, config, name)
        
        # Create task-specific output heads
        self.output_heads = nn.ModuleList()
        for _ in range(num_tasks):
            head = TaskIncrementalBP_layer(
                self.feature_size,
                self.task_output_size,
                Linear()
            )
            self.output_heads.append(head)
    
    def create_network(self, layer_class, activation_fn, out_activation_fn, config):
        _layers = self.orig_layers
        self.layers = nn.ModuleList()
        
        # Create all layers except the last one (which will be task-specific)
        for i in range(len(_layers) - 2):
            self.layers.append(
                layer_class(
                    _layers[i],
                    _layers[i + 1],
                    activation_fn=activation_fn(),
                )
            )
        
        # Note: We don't create the output layer here, as we'll use task-specific heads
    
    def forward(self, x, task_id=None):
        self.input = x
        self.bzs = x.shape[0]
        
        # Pass through shared layers
        for layer in self.layers:
            x = layer(x)
        
        # Features from the shared layers
        features = x
        
        # Use task_id if provided, otherwise use current_task
        if task_id is None:
            task_id = self.current_task
            
        # Pass through task-specific output head
        output = self.output_heads[task_id](features)
        self.y_hat = output
        
        return output
    
    def set_task(self, task_id):
        """Set the current task for the network"""
        self.current_task = task_id
    
    def freeze_previous_tasks(self):
        """Freeze output heads of previously trained tasks"""
        for task_id in self.trained_tasks:
            self.output_heads[task_id].freeze()
    
    def complete_task(self, task_id):
        """Mark a task as completed and freeze its output head"""
        self.trained_tasks.add(task_id)
        self.output_heads[task_id].freeze()
    
    def backward(self, _):
        self.loss.backward()
    
    def get_trainable_parameters(self):
        """Get parameters that should be trained for the current task"""
        params = []
        
        # Always include shared layers
        for layer in self.layers:
            params.extend(list(layer.parameters()))
        
        # Include only unfrozen output heads
        for i, head in enumerate(self.output_heads):
            if not head.is_frozen():
                params.extend(list(head.parameters()))
        
        return params


# Helper functions for Task-IL Split MNIST

def create_split_mnist_tasks(mnist_dataset, num_tasks=5):
    """Split MNIST dataset into multiple tasks"""
    task_datasets = []
    
    for task_id in range(num_tasks):
        # Get indices for digits corresponding to this task
        # For 5 tasks: [0,1], [2,3], [4,5], [6,7], [8,9]
        task_digits = list(range(2 * task_id, 2 * (task_id + 1)))
        
        # Filter indices for the task's digits
        indices = [i for i, (_, label) in enumerate(mnist_dataset) 
                  if label in task_digits]
        
        # Create dataset with remapped labels (0-1)
        task_dataset = [(mnist_dataset[i][0], mnist_dataset[i][1] - 2 * task_id) 
                       for i in indices]
        
        task_datasets.append(task_dataset)
    
    return task_datasets

def train_task_incremental(model, task_datasets, optimizer_class=torch.optim.Adam, 
                          lr=0.001, epochs=10, batch_size=128, device='cuda'):
    """Train the model sequentially on all tasks"""
    accuracies = np.zeros((len(task_datasets), len(task_datasets)))
    
    for task_id, dataset in enumerate(task_datasets):
        print(f"\n--- Training on Task {task_id} ---")
        
        # Set current task
        model.set_task(task_id)
        
        # Freeze previous task output heads
        model.freeze_previous_tasks()
        
        # Create optimizer with only trainable parameters
        optimizer = optimizer_class(model.get_trainable_parameters(), lr=lr)
        
        # Convert dataset to DataLoader
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass with current task
                outputs = model(inputs, task_id)
                loss = model.calculate_loss(outputs, targets)
                
                # Backward pass
                model.backward(None)
                optimizer.step()
                
                running_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            # Print epoch results
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100.0 * correct / total
            print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
        
        # Mark task as completed
        model.complete_task(task_id)
        
        # Evaluate on all seen tasks
        for eval_task_id in range(task_id + 1):
            acc = evaluate_task(model, task_datasets[eval_task_id], eval_task_id, device)
            accuracies[task_id, eval_task_id] = acc
            print(f'Task {eval_task_id} Accuracy after training on Task {task_id}: {acc:.2f}%')
    
    return accuracies

def evaluate_task(model, dataset, task_id, device):
    """Evaluate the model on a specific task"""
    model.eval()
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=128, shuffle=False
    )
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass with task_id
            outputs = model(inputs, task_id)
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100.0 * correct / total
    return accuracy

# Example usage:
"""
# Create a configuration object
from types import SimpleNamespace
config = SimpleNamespace()
config.layers = [784, 256, 256, 2]  # Last layer will be replaced by task-specific heads
config.loss_fn = "ce"
config.device = "cuda" if torch.cuda.is_available() else "cpu"

# Create model
model = TaskIncrementalBP_network(config, num_tasks=5, task_output_size=2)

# Load MNIST dataset
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, download=True, transform=transform)

# Create task datasets
train_tasks = create_split_mnist_tasks(train_dataset)
test_tasks = create_split_mnist_tasks(test_dataset)

# Train model
accuracies = train_task_incremental(model, train_tasks)

# Evaluate final model on all tasks
for task_id in range(5):
    acc = evaluate_task(model, test_tasks[task_id], task_id, config.device)
    print(f'Final accuracy on Task {task_id}: {acc:.2f}%')
"""