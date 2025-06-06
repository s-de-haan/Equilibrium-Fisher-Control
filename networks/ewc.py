import torch
import torch.nn.functional as F
from typing import List

from networks.backprop import BaseNetwork, BaseTaskIncrementalNetwork


class EWCNetwork(BaseNetwork):
    """
    Elastic Weight Consolidation (EWC) Network.
    Implements EWC penalty for continual learning.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, importance: float = 1.0):
        super().__init__(input_dim, hidden_dims, output_dim)
        self.importance = importance
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss with EWC penalty."""
        # Base task loss
        if target.dim() == 2 and target.shape[1] > 1:
            target = target.argmax(dim=1)
        task_loss = F.cross_entropy(output, target)
        
        # Add EWC penalty if not first task
        if not self._first_task:
            ewc_loss = self._compute_ewc_penalty()
            task_loss += ewc_loss
        
        return task_loss
    
    def _compute_ewc_penalty(self) -> torch.Tensor:
        """Compute the EWC penalty term."""
        loss = 0.0
        for n, p in self.named_parameters():
            if n in self._means and n in self._fisher:
                loss += (self._fisher[n] * (p - self._means[n]) ** 2).sum()
        return self.importance * loss
    

class TaskILEWCNetwork(BaseTaskIncrementalNetwork):
    """Task-incremental EWC network."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int], 
                 importance: float = 1.0):
        super().__init__(input_dim, hidden_dims, task_output_dims)
        self.importance = importance
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss with task-incremental EWC penalty."""
        # Base task loss
        if target.dim() == 2 and target.shape[1] > 1:
            target = target.argmax(dim=1)
        task_loss = F.cross_entropy(output, target)
        
        # Add EWC penalty for previous tasks
        if not self._first_task:
            ewc_loss = self._compute_task_ewc_penalty()
            task_loss += ewc_loss
        
        return task_loss
    
    def _compute_task_ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty for all previous tasks."""
        loss = 0.0
        
        # Apply penalty for each previous task
        for task_id in range(self.current_task):
            if task_id not in self.task_fisher:
                continue
            
            task_loss = 0.0
            for n, p in self.named_parameters():
                # Skip current task's output head
                if f"output_heads.{self.current_task}" in n:
                    continue
                
                if n in self.task_fisher[task_id] and n in self.task_means[task_id]:
                    task_loss += (self.task_fisher[task_id][n] * (p - self.task_means[task_id][n]) ** 2).sum()
            
            loss += task_loss
        
        return self.importance * loss
