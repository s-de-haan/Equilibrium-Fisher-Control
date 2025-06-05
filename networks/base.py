import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union, Tuple


class BaseNetwork(nn.Module, ABC):
    """
    Unified base class for all network types.
    Provides consistent interface that works with UnifiedWandBTrainer.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.all_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Build the network
        self.layers = nn.ModuleList()
        for i in range(len(self.all_dims) - 1):
            self.layers.append(nn.Linear(self.all_dims[i], self.all_dims[i + 1]))
        
        # Default activation
        self.activation = nn.ReLU()
        
        # Initialize continual learning state
        self._fisher = {}
        self._means = {}
        self._first_task = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for inference."""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass. Default implementation is same as forward.
        Override in subclasses for special training behavior (like EFC).
        """
        return self.forward(x)
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss (default is cross-entropy)."""
        if target.dim() == 2 and target.shape[1] > 1:
            # One-hot encoded targets
            target = target.argmax(dim=1)
        return F.cross_entropy(output, target)
    
    def backward(self):
        """
        Custom backward pass. Default is empty (uses autograd).
        Override in subclasses if needed.
        """
        pass
    
    def complete_task(self, dataloader, device=None):
        """
        Complete a task. Default implementation computes Fisher information.
        Override in subclasses for different behavior.
        """
        if device is None:
            device = next(self.parameters()).device
            
        # Compute Fisher Information Matrix
        fisher = self._compute_fisher_information(dataloader, device)
        
        # Store current parameters
        means = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}
        
        if self._first_task:
            self._fisher = fisher
            self._means = means
            self._first_task = False
        else:
            # Accumulate Fisher information
            for n in self._fisher:
                if n in fisher:
                    self._fisher[n] += fisher[n]
            self._means = means
    
    def _compute_fisher_information(self, dataloader, device):
        """Compute Fisher Information Matrix."""
        fisher = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)
        
        self.eval()
        num_samples = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = self(inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # Convert targets to proper format
            if targets.dim() == 2:  # one-hot
                log_likelihood = (log_probs * targets).sum(dim=1)
            else:  # class indices
                log_likelihood = log_probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            # Compute gradients
            self.zero_grad()
            log_likelihood.sum().backward()
            
            # Accumulate squared gradients
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            
            num_samples += inputs.size(0)
        
        # Normalize by dataset size
        for n in fisher:
            fisher[n] /= num_samples
        
        return fisher


class TaskIncrementalMixin:
    """
    Mixin for task-incremental learning capabilities.
    Add this to any BaseNetwork subclass to make it task-incremental.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int]):
        # Initialize the base network (feature extractor)
        super().__init__(input_dim, hidden_dims[:-1], hidden_dims[-1])
        
        # Remove the output layer from base network
        self.layers = self.layers[:-1]
        
        # Create task-specific output heads
        self.output_heads = nn.ModuleList()
        for output_dim in task_output_dims:
            self.output_heads.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # Task management
        self.current_task = 0
        self.trained_tasks = set()
        self.task_fisher = {}  # Task-specific Fisher matrices
        self.task_means = {}   # Task-specific parameter means
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Forward pass with task-specific output head."""
        if task_id is None:
            task_id = self.current_task
        
        # Pass through shared feature extractor
        for layer in self.layers:
            x = self.activation(layer(x))
        
        # Pass through task-specific output head
        return self.output_heads[task_id](x)
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Training forward pass with task ID."""
        if task_id is None:
            task_id = self.current_task
        return self.forward(x, task_id)
    
    def set_task(self, task_id: int):
        """Set the current task."""
        self.current_task = task_id
    
    def freeze_previous_tasks(self):
        """Freeze output heads of previously trained tasks."""
        for task_id in self.trained_tasks:
            for param in self.output_heads[task_id].parameters():
                param.requires_grad = False
    
    def complete_task_and_freeze_output_head(self, task_id: int):
        """Mark task as completed and freeze its output head."""
        self.trained_tasks.add(task_id)
        for param in self.output_heads[task_id].parameters():
            param.requires_grad = False
    
    def complete_task(self, dataloader, device=None):
        """Task-incremental version of complete_task."""
        if device is None:
            device = next(self.parameters()).device
        
        # Compute Fisher for current task
        fisher = self._compute_fisher_information(dataloader, device)
        means = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}
        
        # Store task-specific Fisher and means
        self.task_fisher[self.current_task] = fisher
        self.task_means[self.current_task] = means
        
        if self._first_task:
            self._fisher = fisher
            self._means = means
            self._first_task = False


class BaseTaskIncrementalNetwork(TaskIncrementalMixin, BaseNetwork):
    """Base class for task-incremental networks."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int]):
        TaskIncrementalMixin.__init__(self, input_dim, hidden_dims, task_output_dims)
        
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Task-incremental loss computation."""
        if target.dim() == 2 and target.shape[1] > 1:
            target = target.argmax(dim=1)
        return F.cross_entropy(output, target)