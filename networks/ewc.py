import torch
from typing import List

from networks.backprop import BP_Network, TaskIL_BP_Network

class EWC_Network(BP_Network):
    """
    Elastic Weight Consolidation (EWC) Network.
    Implements EWC penalty for continual learning.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, importance: float = 1.0):
        """
        Initialize an EWC network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            importance: Importance coefficient for EWC penalty
        """
        super().__init__(input_dim, hidden_dims, output_dim)
        self.importance = importance
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass - for EWC, this is the same as the standard forward pass.
        The EWC penalty is added during loss computation.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        return self.forward(x)
    
    def ewc_loss(self) -> torch.Tensor:
        """
        Compute the EWC penalty term.
        
        Returns:
            torch.Tensor: EWC penalty
        """
        if self._first_task:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self._means and n in self._fisher:
                loss += (self._fisher[n] * (p - self._means[n])**2).sum()
                
        return self.importance * loss
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with EWC penalty.
        
        Args:
            output: Model output
            target: Target values
            
        Returns:
            torch.Tensor: Loss with EWC penalty
        """
        task_loss = super().compute_loss(output, target)
        
        if not self._first_task:
            task_loss += self.ewc_loss()
            
        return task_loss
    
    def backward(self):
        pass


class TaskIL_EWC_Network(TaskIL_BP_Network):
    """
    Task-incremental learning version of EWC network.
    Maintains separate output heads for each task and applies EWC to shared parameters.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int], importance: float = 1.0):
        """
        Initialize a task-incremental EWC network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            task_output_dims: List of output dimensions for each task
            importance: Importance coefficient for EWC penalty
        """
        super().__init__(input_dim, hidden_dims, task_output_dims)
        self.importance = importance
        self.task_fisher = {}  # Fisher matrix per task
        self.task_means = {}   # Parameter means per task
        
    def ewc_loss(self) -> torch.Tensor:
        """
        Compute the EWC penalty term for task-incremental learning.
        Applies EWC only to shared parameters.
        
        Returns:
            torch.Tensor: EWC penalty
        """
        if self._first_task:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        loss = 0.0
        
        # Apply EWC penalty for each previously seen task
        for task_id in range(self.current_task):
            if task_id not in self.task_fisher:
                continue
                
            task_loss = 0.0
            for n, p in self.named_parameters():
                # Only apply to shared parameters or parameters of previous tasks
                if n not in self.task_fisher[task_id] or n not in self.task_means[task_id]:
                    continue
                    
                # Skip current task's output head parameters
                if f"output_heads.{self.current_task}" in n:
                    continue
                    
                task_loss += (self.task_fisher[task_id][n] * (p - self.task_means[task_id][n])**2).sum()
                
            loss += task_loss
                
        return self.importance * loss
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with EWC penalty.
        
        Args:
            output: Model output
            target: Target values
            
        Returns:
            torch.Tensor: Loss with EWC penalty
        """
        task_loss = super().compute_loss(output, target)
        
        if not self._first_task:
            task_loss += self.ewc_loss()
            
        return task_loss
    
    def complete_task(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
        """
        Complete a task and compute task-specific Fisher matrix.
        
        Args:
            dataloader: DataLoader for the task
            device: Device to use for computation
        """
        # Compute the Fisher matrix for the current task
        current_fisher = self.calculate_fisher(dataloader, device)
        current_means = {n: p.data.clone() for n, p in self.named_parameters()}
        
        # Store in task-specific dictionaries
        self.task_fisher[self.current_task] = current_fisher
        self.task_means[self.current_task] = current_means
        
        # Update first_task flag
        if self._first_task:
            self._first_task = False
            
        # Increment task ID if using automatic task progression
        # self.current_task += 1  # Uncomment if automatic progression is desired

    def backward(self):
        pass