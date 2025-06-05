import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from networks.base import BaseTaskIncrementalNetwork, BaseNetwork


class DFCNetwork(BaseNetwork):
    """
    Deep Feedback Control (DFC) Network.
    Implements biologically plausible credit assignment.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dt: float = 0.1, tau: float = 0.1, k_p: float = 1.0, 
                 alpha: float = 0.1, max_iter: int = 50, tol: float = 1e-4):
        super().__init__(input_dim, hidden_dims, output_dim)
        
        # DFC parameters
        self.dt = dt
        self.tau = tau
        self.k_p = k_p
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
        # Use tanh for better stability
        self.activation = nn.Tanh()
        
        # Feedback weights (B matrices in DFC paper)
        self.feedback_weights = nn.ParameterList()
        for i in range(len(self.layers) - 1):
            self.feedback_weights.append(
                nn.Parameter(torch.randn(self.all_dims[i+1], output_dim) / (output_dim ** 0.5))
            )
        # Output layer feedback is identity
        self.feedback_weights.append(nn.Parameter(torch.eye(output_dim)))
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Training forward pass with dynamic inversion."""
        # Convert targets to one-hot if needed
        if y.dim() == 1:
            y = F.one_hot(y, self.output_dim).float()
        
        # Perform dynamic inversion to find equilibrium
        equilibrium_acts = self._dynamic_inversion(x, y)
        
        return equilibrium_acts[-1]
    
    def _dynamic_inversion(self, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        """Perform dynamic inversion to find steady-state activations."""
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize activations with feedforward pass
        activations = [x]
        for i, layer in enumerate(self.layers[:-1]):
            activations.append(self.activation(layer(activations[-1])))
        # Output layer (no activation)
        activations.append(self.layers[-1](activations[-1]))
        
        # Initialize control signals
        u = torch.zeros_like(y)
        u_int = torch.zeros_like(y)
        errors = [torch.zeros_like(act) for act in activations[1:]]
        
        # Dynamic inversion loop
        for t in range(self.max_iter):
            # Compute prediction error
            pred_error = activations[-1] - y
            
            # PI controller
            u_int = u_int + self.dt * (pred_error - self.alpha * u)
            u_new = u_int + self.k_p * pred_error
            
            # Check convergence
            if torch.norm(u_new - u) < self.tol:
                break
            u = u_new
            
            # Update activations with feedback
            new_activations = [x]
            
            for i in range(len(self.layers)):
                # Forward computation
                if i < len(self.layers) - 1:
                    forward_output = self.activation(self.layers[i](new_activations[i]))
                else:
                    forward_output = self.layers[i](new_activations[i])
                
                # Feedback control
                feedback = torch.matmul(u, self.feedback_weights[i].t())
                
                # Update error
                errors[i] = errors[i] + self.dt/self.tau * (feedback - errors[i])
                
                # Update activation
                if i < len(self.layers) - 1:
                    next_act = activations[i+1] + self.dt/self.tau * (forward_output + errors[i] - activations[i+1])
                else:
                    next_act = activations[i+1] + self.dt/self.tau * (forward_output + errors[i] - activations[i+1])
                
                new_activations.append(next_act)
            
            activations = new_activations
        
        return activations
    

class TaskILDFCNetwork(BaseTaskIncrementalNetwork):
    """Task-incremental DFC network."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int],
                 dt: float = 0.1, tau: float = 0.1, k_p: float = 1.0, 
                 alpha: float = 0.1, max_iter: int = 50, tol: float = 1e-4):
        super().__init__(input_dim, hidden_dims, task_output_dims)
        
        # DFC parameters
        self.dt = dt
        self.tau = tau
        self.k_p = k_p
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        
        # Use tanh for better stability
        self.activation = nn.Tanh()
        
        # Task-specific feedback weights
        self.feedback_weights = nn.ParameterList()
        for i in range(len(self.layers)):
            self.feedback_weights.append(
                nn.Parameter(torch.randn(self.all_dims[i+1], hidden_dims[-1]) / (hidden_dims[-1] ** 0.5))
            )
        
        # Task-specific output feedback
        self.output_feedback_weights = nn.ParameterList()
        for output_dim in task_output_dims:
            self.output_feedback_weights.append(
                nn.Parameter(torch.randn(hidden_dims[-1], output_dim) / (output_dim ** 0.5))
            )
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Task-specific training forward pass with dynamic inversion."""
        if task_id is None:
            task_id = self.current_task
        
        # Convert targets to one-hot if needed
        if y.dim() == 1:
            output_dim = self.output_heads[task_id].out_features
            y = F.one_hot(y, output_dim).float()
        
        # Perform task-specific dynamic inversion
        equilibrium_acts = self._task_dynamic_inversion(x, y, task_id)
        
        return equilibrium_acts[-1]
    
    def _task_dynamic_inversion(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> List[torch.Tensor]:
        """Task-specific dynamic inversion."""
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize activations
        activations = [x]
        for layer in self.layers:
            activations.append(self.activation(layer(activations[-1])))
        # Task-specific output
        activations.append(self.output_heads[task_id](activations[-1]))
        
        # Initialize control signals
        u = torch.zeros_like(y)
        u_int = torch.zeros_like(y)
        errors = [torch.zeros_like(act) for act in activations[1:]]
        
        # Dynamic inversion loop
        for t in range(self.max_iter):
            pred_error = activations[-1] - y
            
            # PI controller
            u_int = u_int + self.dt * (pred_error - self.alpha * u)
            u_new = u_int + self.k_p * pred_error
            
            if torch.norm(u_new - u) < self.tol:
                break
            u = u_new
            
            # Update shared layers
            new_activations = [x]
            for i in range(len(self.layers)):
                forward_output = self.activation(self.layers[i](new_activations[i]))
                
                # Task-specific feedback
                feedback = torch.matmul(u, self.output_feedback_weights[task_id].t())
                if i < len(self.layers) - 1:
                    feedback = torch.matmul(feedback, self.feedback_weights[i].t())
                
                errors[i] = errors[i] + self.dt/self.tau * (feedback - errors[i])
                next_act = activations[i+1] + self.dt/self.tau * (forward_output + errors[i] - activations[i+1])
                new_activations.append(next_act)
            
            # Update task-specific output
            forward_output = self.output_heads[task_id](new_activations[-1])
            errors[-1] = errors[-1] + self.dt/self.tau * (u - errors[-1])
            next_act = activations[-1] + self.dt/self.tau * (forward_output + errors[-1] - activations[-1])
            new_activations.append(next_act)
            
            activations = new_activations
        
        return activations