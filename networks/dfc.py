import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any

from networks.base import EquilibriumModule
from networks.fixedpoint_solver import get_solver


class DFC_Linear(nn.Linear):
    """Custom linear layer with DFC-specific attributes and methods"""
    
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.activation_fn = nn.Softplus(beta=5)
        self.r_prev = None
        self.v_ff = None
        self.r_ff = None
        self.r = None
    
    def activation_derivative(self, x):
        """Compute derivative of softplus with beta=5"""
        return torch.sigmoid(5 * x)
    
    def forward(self, x):
        self.r_prev = x
        self.v_ff = F.linear(x, self.weight, self.bias)
        self.r_ff = self.activation_fn(self.v_ff)
        self.r = self.r_ff.clone()
        return self.r


class DFC_Network(EquilibriumModule):
    """
    Deep Feedback Control Network.
    Implements biologically plausible credit assignment with dynamic inversion.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, config: Any):
        """
        Initialize a DFC network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            config: Configuration object with DFC parameters
        """
        super().__init__(input_dim, hidden_dims, output_dim)
        
        # Store DFC parameters
        self.dt = getattr(config, 'dt', 0.1)
        self.tau_v = getattr(config, 'tau_v', 0.1)
        self.tau_u = getattr(config, 'tau_u', 0.1)
        self.tau_e = getattr(config, 'tau_e', 0.1)
        self.k_p = getattr(config, 'k_p', 1.0)
        self.k_i = getattr(config, 'k_i', 0.5)
        self.alpha = getattr(config, 'alpha', 0.1)
        self.max_iter = getattr(config, 'max_iter', 50)
        self.tol = getattr(config, 'tol', 1e-4)
        
        # Use tanh activation for better stability in dynamic inversion
        self.activation = nn.Tanh()
        
        # Create feedback pathways (B matrices in the DFC paper)
        self.feedback_weights = nn.ParameterList()
        for i in range(len(self.layers) - 1):
            self.feedback_weights.append(
                nn.Parameter(torch.randn(self.all_dims[i+1], output_dim) / (output_dim ** 0.5))
            )
        # Last layer feedback is identity (direct teaching signal)
        self.feedback_weights.append(nn.Parameter(torch.eye(output_dim)))
        
        # Create solver for fixed-point computations
        self.solver = get_solver(config)

    def activation_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute derivative of activation function.
        For tanh, it's 1 - tanh(x)^2
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Derivative of activation at input points
        """
        return 1 - torch.tanh(x)**2
    
    def calculate_jacobian_products(self, activations: List[torch.Tensor], control_signal: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculate the product of Jacobian matrices and control signal for each layer.
        This is an efficient way to compute J^T u without materializing the full Jacobian.
        
        Args:
            activations: List of activations at each layer
            control_signal: Error feedback signal
            
        Returns:
            List[torch.Tensor]: Jacobian-vector products for each layer
        """
        psi_list = []
        
        # For output layer, Jacobian is identity since no activation function
        psi = control_signal  # J^T u = u for output layer
        psi_list.append(psi)
        
        # Backward propagation of error signal through the network
        for i in range(len(self.layers)-2, -1, -1):
            # Pre-activation values for current layer
            pre_act = self.layers[i](activations[i])
            
            # Compute J^T u for current layer via chain rule
            # J_{i} = diag(σ'(pre_act)) W_{i+1}^T
            act_deriv = self.activation_derivative(pre_act)
            psi = torch.matmul(psi, self.layers[i+1].weight) * act_deriv
            
            # Store the result
            psi_list.insert(0, psi)  # Insert at beginning to maintain layer order
            
        return psi_list
    
    def dynamic_inversion(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform dynamic inversion to find steady-state activations.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Layer activations and control signals
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states
        activations = [x]  # First activation is the input
        for i, layer in enumerate(self.layers[:-1]):
            # Initialize with forward pass
            activations.append(self.activation(layer(activations[-1])))
        # Output layer (no activation)
        activations.append(self.layers[-1](activations[-1]))
        
        # Initialize control signals and errors
        u = torch.zeros_like(activations[-1])
        u_int = torch.zeros_like(u)
        errors = [torch.zeros_like(act) for act in activations[1:]]  # No error for input layer
        
        # Create a mask to track convergence
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Perform dynamic inversion
        for t in range(self.max_iter):
            # Compute prediction error
            pred_error = activations[-1] - y
            
            # PI controller for output layer
            u_int = u_int + self.dt * (pred_error - self.alpha * u)
            u = u_int + self.k_p * pred_error
            
            # Check convergence
            converged = converged | (torch.norm(u, dim=1) < self.tol)
            if converged.all():
                break
            
            # # Calculate Jacobian-vector products
            # psi_list = self.calculate_jacobian_products(activations, u)
            
            # Update activations using control signals
            new_activations = [x]  # Input remains unchanged
            
            for i in range(len(self.layers)):
                # Forward computation
                forward_input = new_activations[i]
                forward_output = self.layers[i](forward_input)
                
                # Apply activation function for hidden layers
                if i < len(self.layers) - 1:
                    forward_output = self.activation(forward_output)
                
                # Apply feedback control
                feedback = torch.matmul(u, self.feedback_weights[i].t())
                
                # Update error using feedback alignment
                errors[i] = errors[i] + self.dt/self.tau_e * (feedback - errors[i])
                
                # Update activation with error
                if i < len(self.layers) - 1:
                    next_act = activations[i+1] + self.dt/self.tau_v * (forward_output + errors[i] - activations[i+1])
                else:
                    next_act = activations[i+1] + self.dt/self.tau_v * (forward_output + errors[i] - activations[i+1] + u)
                
                new_activations.append(next_act)
            
            # Update activations for next iteration
            activations = new_activations
            
        return activations, errors
        
    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass with dynamic inversion.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Perform dynamic inversion to find steady-state activations and errors
        activations, errors = self.dynamic_inversion(x, y)
        
        # Extract output activations
        output = activations[-1]
        
        # Register hooks for custom backward pass
        for i in range(1, len(activations)):
            def get_backward_hook(i, error):
                def backward_hook(grad):
                    # For DFC, we replace the gradient with the error signal
                    return -error
                return backward_hook
            
            activations[i].register_hook(get_backward_hook(i, errors[i-1]))
            
        return output
    

class TaskIL_DFC_Network(DFC_Network):
    """
    Task-incremental learning version of DFC network.
    Maintains separate output heads for each task.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int], config: Any):
        """
        Initialize a task-incremental DFC network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            task_output_dims: List of output dimensions for each task
            config: Configuration object with DFC parameters
        """
        # Initialize with shared feature extractor (minus the output layer)
        super().__init__(input_dim, hidden_dims[:-1], hidden_dims[-1], config)
        
        # Remove the last layer from the main network
        self.layers = self.layers[:-1]
        
        # Create task-specific output heads
        self.output_heads = nn.ModuleList()
        for task_dim in task_output_dims:
            self.output_heads.append(nn.Linear(hidden_dims[-1], task_dim))
            
        # Create task-specific feedback matrices
        self.feedback_weights = nn.ParameterList()
        for i in range(len(self.layers)):
            self.feedback_weights.append(
                nn.Parameter(torch.randn(self.all_dims[i+1], hidden_dims[-1]) / (hidden_dims[-1] ** 0.5))
            )
            
        # Create task-specific feedback from output heads to hidden layer
        self.output_feedback_weights = nn.ParameterList()
        for task_dim in task_output_dims:
            self.output_feedback_weights.append(
                nn.Parameter(torch.randn(hidden_dims[-1], task_dim) / (task_dim ** 0.5))
            )
            
        self.current_task = 0
        
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
    
    def dynamic_inversion_task_specific(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform dynamic inversion for task-specific output head.
        
        Args:
            x: Input tensor
            y: Target tensor
            task_id: Task ID
            
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Layer activations and control signals
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states
        activations = [x]  # First activation is the input
        for i, layer in enumerate(self.layers):
            # Initialize with forward pass
            activations.append(self.activation(layer(activations[-1])))
        # Output layer (task-specific, no activation)
        activations.append(self.output_heads[task_id](activations[-1]))
        
        # Initialize control signals and errors
        u = torch.zeros_like(activations[-1])
        u_int = torch.zeros_like(u)
        errors = [torch.zeros_like(act) for act in activations[1:]]  # No error for input layer
        
        # Create a mask to track convergence
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Perform dynamic inversion
        for t in range(self.max_iter):
            # Compute prediction error
            pred_error = activations[-1] - y
            
            # PI controller for output layer
            u_int = u_int + self.dt * (pred_error - self.alpha * u)
            u = u_int + self.k_p * pred_error
            
            # Check convergence
            converged = converged | (torch.norm(u, dim=1) < self.tol)
            if converged.all():
                break
            
            # Update activations using control signals
            new_activations = [x]  # Input remains unchanged
            
            # Shared layers
            for i in range(len(self.layers)):
                # Forward computation
                forward_input = new_activations[i]
                forward_output = self.activation(self.layers[i](forward_input))
                
                # Apply feedback control (using the shared feedback pathway)
                feedback = torch.matmul(u, self.output_feedback_weights[task_id].t())
                if i < len(self.layers) - 1:
                    feedback = torch.matmul(feedback, self.feedback_weights[i].t())
                
                # Update error using feedback alignment
                errors[i] = errors[i] + self.dt/self.tau_e * (feedback - errors[i])
                
                # Update activation with error
                next_act = activations[i+1] + self.dt/self.tau_v * (forward_output + errors[i] - activations[i+1])
                new_activations.append(next_act)
            
            # Output layer (task-specific)
            forward_output = self.output_heads[task_id](new_activations[-1])
            errors[-1] = errors[-1] + self.dt/self.tau_e * (u - errors[-1])
            next_act = activations[-1] + self.dt/self.tau_v * (forward_output + errors[-1] - activations[-1] + u)
            new_activations.append(next_act)
            
            # Update activations for next iteration
            activations = new_activations
            
        return activations, errors
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Training forward pass with dynamic inversion for task-specific head.
        
        Args:
            x: Input tensor
            y: Target tensor
            task_id: Task ID (optional, uses current_task if None)
            
        Returns:
            torch.Tensor: Output predictions
        """
        if task_id is None:
            task_id = self.current_task
            
        # Perform dynamic inversion to find steady-state activations and errors
        activations, errors = self.dynamic_inversion_task_specific(x, y, task_id)
        
        # Extract output activations
        output = activations[-1]
        
        # Register hooks for custom backward pass
        for i in range(1, len(activations)):
            def get_backward_hook(i, error):
                def backward_hook(grad):
                    # For DFC, we replace the gradient with the error signal
                    return -error
                return backward_hook
            
            activations[i].register_hook(get_backward_hook(i, errors[i-1]))
            
        return output
    
    def set_task(self, task_id: int) -> None:
        """Set the current task ID."""
        self.current_task = task_id