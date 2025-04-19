import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import List, Dict

class EquilibriumModule(nn.Module):
    """
    Base class for all equilibrium-based modules.
    This provides a clean interface for various learning algorithms.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        """
        Initialize the equilibrium module.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
        """
        super().__init__()
        
        # Store architecture dimensions
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.all_dims = [input_dim] + hidden_dims + [output_dim]
        
        # Create standard PyTorch layers
        self.layers = nn.ModuleList()
        for i in range(len(self.all_dims) - 1):
            self.layers.append(nn.Linear(self.all_dims[i], self.all_dims[i + 1]))
            
        # Default activation function
        self.activation = nn.ReLU()
        
        # Initialize Fisher matrix for continual learning
        self._fisher = {}
        self._means = {}
        self._first_task = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard feedforward pass for inference."""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
        return self.layers[-1](x)
    
    def activations(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Get all layer activations (including input)."""
        activations = [x]
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            activations.append(x)
        x = self.layers[-1](x)
        activations.append(x)
        return activations
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss (default is cross-entropy)."""
        # Check if target is one-hot encoded
        if target.dim() > 1 and target.shape[1] > 1:
            # Convert one-hot to class indices
            target = target.argmax(dim=1)
        return F.cross_entropy(output, target)

    def calculate_fisher(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix (for EWC and EFC)."""
        fisher = {}
        for n, p in self.named_parameters():
            fisher[n] = torch.zeros_like(p)
            
        self.eval()
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = self(inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # Can be modified for one-hot targets if needed
            if targets.dim() == 2:  # one-hot
                log_likelihood = (log_probs * targets).sum(dim=1)
            else:  # class indices
                log_likelihood = log_probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            # Compute gradients
            self.zero_grad()
            log_likelihood.sum().backward()
            
            # Accumulate squared gradients in Fisher
            for n, p in self.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
                    
        # Normalize by dataset size
        for n in fisher.keys():
            fisher[n] /= len(dataloader.dataset)
            
        return fisher
    
    @abstractmethod
    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Training forward pass, to be implemented by specific algorithms."""
        pass