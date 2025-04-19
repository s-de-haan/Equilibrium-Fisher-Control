import torch
import torch.nn as nn
from typing import List, Optional

from networks.base import EquilibriumModule

class BP_Network(EquilibriumModule):
    """
    Standard backpropagation neural network.
    A simple implementation without any special requirements for training.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Initialize a standard backpropagation network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
        """
        super().__init__(input_dim, hidden_dims, output_dim)
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass - for BP, this is the same as the standard forward pass.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        # For BP, training forward is the same as standard forward
        return self.forward(x)
    
    def backward(self):
        pass


# Task-Incremental BP Network for Split-MNIST style tasks
class TaskIL_BP_Network(BP_Network):
    """
    Task-incremental learning version of backpropagation network.
    Maintains separate output heads for each task.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int]):
        """
        Initialize a task-incremental BP network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            task_output_dims: List of output dimensions for each task
        """
        # Initialize with feature extractor only (minus the output layer)
        super().__init__(input_dim, hidden_dims[:-1], hidden_dims[-1])
        
        # Remove the last layer from the main network
        self.layers = self.layers[:-1]
        
        # Create task-specific output heads
        self.output_heads = nn.ModuleList()
        for task_dim in task_output_dims:
            self.output_heads.append(nn.Linear(hidden_dims[-1], task_dim))
            
        self.current_task = 0
        self.task_masks = {}  # For EWC with task-specific parameters
        
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass using the appropriate task head.
        
        Args:
            x: Input tensor
            task_id: Task ID (optional, uses current_task if None)
            
        Returns:
            torch.Tensor: Output predictions for the specified task
        """
        if task_id is None:
            task_id = self.current_task
            
        # Pass through shared layers
        for layer in self.layers:
            x = self.activation(layer(x))
            
        # Pass through task-specific output head
        return self.output_heads[task_id](x)
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Training forward pass - for BP, this is the same as the standard forward pass.
        
        Args:
            x: Input tensor
            y: Target tensor
            task_id: Task ID (optional, uses current_task if None)
            
        Returns:
            torch.Tensor: Output predictions
        """
        return self.forward(x, task_id)
    
    def set_task(self, task_id: int) -> None:
        """Set the current task ID."""
        self.current_task = task_id
    
    def get_task_parameters(self, task_id: int) -> List[nn.Parameter]:
        """
        Get parameters specific to a task.
        For BP with task-incremental learning, this is the output head parameters.
        
        Args:
            task_id: Task ID
            
        Returns:
            List[nn.Parameter]: List of parameters specific to the task
        """
        return list(self.output_heads[task_id].parameters())
    
    def get_shared_parameters(self) -> List[nn.Parameter]:
        """
        Get parameters shared across all tasks.
        For BP with task-incremental learning, these are the feature extractor parameters.
        
        Returns:
            List[nn.Parameter]: List of shared parameters
        """
        return list(self.layers.parameters())