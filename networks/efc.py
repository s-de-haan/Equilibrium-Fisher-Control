import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Any, Dict

from networks.dfc import DFC_Linear, DFC_Network, TaskIL_DFC_Network

class EFC_Network(DFC_Network):
    """
    Equilibrium Fisher Control Network.
    Extends DFC with Fisher-based modulation for continual learning.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, config: Any):
        """
        Initialize an EFC network.
        """
        super().__init__(input_dim, hidden_dims, output_dim, config)
        
        # EFC parameters
        self.beta = getattr(config, 'beta_efc', 1.0)
        
        # Activation function with beta=5 for stability
        self.activation = nn.Softplus(beta=5)
        
        # Dynamic inversion parameters
        self.dt = getattr(config, 'dt_di', 0.008)
        self.time_constant_ratio = getattr(config, 'time_constant_ratio', 0.2)
        self.k_p = getattr(config, 'k_p', 2.0)
        self.tmax = getattr(config, 'tmax_di', 500)
        self.eps = getattr(config, 'eps', 1e-4)
        self.alpha = getattr(config, 'alpha_di', 1e-4)
        self.target_lr = getattr(config, 'target_lr', 0.01)
        
        # Layer-specific time constants
        self.tau = getattr(config, 'taus', [self.dt/self.time_constant_ratio] * len(self.layers))
        
        # Fisher information storage
        self._first_task = True
        self._fisher = {}
        self._means = {}
        
        # Initialize softmax for CE loss
        self._softmax = nn.Softmax(dim=1)

        # Create network with custom layers
        self.layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims + [output_dim]
        
        for i in range(len(dims) - 1):
            self.layers.append(DFC_Linear(dims[i], dims[i+1]))
    
    def _set_targets_ce(self, y):
        """Process targets for CE loss with dimension checks"""
        # Convert class indices to one-hot if needed
        if y.dim() == 1:
            y_one_hot = torch.zeros(y.size(0), self.layers[-1].out_features, device=y.device)
            y_one_hot.scatter_(1, y.unsqueeze(1), 1)
            y = y_one_hot
        
        # Ensure y_hat and y have the same batch size
        self.targets = self._softmax(self.y_hat) - self.target_lr * (self._softmax(self.y_hat) - y)
        
        self.output_size = self.targets.shape[1]

    def _set_targets_mse(self, y):
        """Process targets for MSE loss with dimension checks"""
        # Ensure y_hat and y have the same batch size
        if self.y_hat.size(0) != y.size(0):
            # Handle the case where batch sizes don't match
            min_size = min(self.y_hat.size(0), y.size(0))
            y_hat_subset = self.y_hat[:min_size]
            y_subset = y[:min_size]
            
            # Update self.y_hat to maintain consistency
            self.y_hat = y_hat_subset
            
            # Compute targets with matching dimensions
            self.targets = (1 - 2 * self.target_lr) * y_hat_subset + 2 * self.target_lr * y_subset
        else:
            # Normal case with matching dimensions
            self.targets = (1 - 2 * self.target_lr) * self.y_hat + 2 * self.target_lr * y
        
        self.output_size = self.targets.shape[1]

    def _set_targets(self, y):
        """Process targets based on loss function with safety checks"""
        # First, ensure y_hat is initialized and has the correct shape
        if not hasattr(self, 'y_hat') or self.y_hat is None:
            raise ValueError("y_hat must be initialized before setting targets")
        
        # Check and fix dimension issues
        if y.size(0) != self.y_hat.size(0):
            print(f"Warning: Target batch size ({y.size(0)}) doesn't match y_hat batch size ({self.y_hat.size(0)})")
        
        # Choose method based on loss function
        loss_fn = getattr(self, 'loss_fn_name', 'ce')
        if loss_fn == 'mse':
            self._set_targets_mse(y)
        else:
            self._set_targets_ce(y)

    def _compute_error_mse(self, y_hat, y):
        """Compute error for MSE loss"""
        return y - y_hat

    def _compute_error_ce(self, y_hat, y):
        """Compute error for CE loss"""
        return y - self._softmax(y_hat)
    
    def _compute_error(self, y_hat, y):
        """Compute prediction error based on loss function"""
        loss_fn = getattr(self, 'loss_fn_name', 'ce')
        if loss_fn == 'mse':
            return self._compute_error_mse(y_hat, y)
        else:
            return self._compute_error_ce(y_hat, y)
    
    def _calculate_psis(self, u):
        """Calculate control signals for each layer from output control signal"""
        L = len(self.layers)
        psi_list = [None] * L

        # Get activation derivatives for each layer
        activations_derivatives = [layer.activation_derivative(layer.v_ff) for layer in self.layers]
        
        # Last layer
        psi = u * activations_derivatives[-1]
        psi_list[-1] = psi
        
        # Backpropagate from second-to-last to first layer
        for i in range(L - 2, -1, -1):
            psi = (psi @ self.layers[i + 1].weight) * activations_derivatives[i]
            psi_list[i] = psi
        
        return psi_list
    
    @torch.no_grad()
    def dynamic_inversion(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform dynamic inversion with Fisher modulation based on EFC_network_v5.
        """
        # Store input and initialize
        self.input = x
        self.bzs = x.shape[0]
        
        # Forward pass to get initial y_hat if not yet available
        if not hasattr(self, 'y_hat') or self.y_hat is None:
            self.y_hat = self.forward(x)
        
        # Set targets based on the current y_hat and loss function
        self._set_targets(y)
        
        # Initialize activations for all layers
        activations = [x]  # Input is first activation
        for i, layer in enumerate(self.layers):
            # Initialize with forward computation
            layer.r_prev = activations[-1]
            layer.v_ff = layer.r_prev @ layer.weight.T + layer.bias
            layer.r_ff = self.activation(layer.v_ff)
            layer.r = layer.r_ff.clone()  # Start with feedforward activations
            activations.append(layer.r.clone().detach().requires_grad_(True))  # Make it require gradients
        
        # Initialize control signal and convergence mask
        u_current = torch.zeros((self.bzs, self.output_size), device=x.device)
        u_int = torch.zeros((self.bzs, self.output_size), device=x.device)
        errors = [torch.zeros_like(act) for act in activations[1:]]  # No error for input
        converged_mask = torch.zeros(self.bzs, dtype=torch.bool, device=x.device)
        
        # Dynamic inversion loop
        for t in range(self.tmax):
            # Compute prediction error
            error = self._compute_error(activations[-1], self.targets)
            
            # PI controller
            u_int = u_int + self.dt * (error - self.alpha * u_current)
            u_next = u_int + self.k_p * error
            
            # Check convergence
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all():
                break
            
            # Calculate psi values for each layer
            psis = self._calculate_psis(u_next)
            
            # Update each layer's activation
            for i, layer in enumerate(self.layers):
                # Forward computation for up-to-date r_prev
                layer.r_prev = activations[i]
                layer.v_ff = layer.r_prev @ layer.weight.T + layer.bias
                layer.r_ff = self.activation(layer.v_ff)
                
                # Compute modulation factor
                psi = psis[i]
                e_psi_gamma = torch.tanh(psi) + 1  # Simple modulation from v5
                
                # Update activation with modulation
                tau = self.tau[i] if isinstance(self.tau, list) else self.dt/self.time_constant_ratio
                layer.r = layer.r + tau * (e_psi_gamma * layer.r_ff - layer.r)
                
                # Calculate error for backward pass
                errors[i] = layer.r - layer.r_ff
                
                # Update stored activations with new ones that require gradients
                activations[i+1] = layer.r.clone().detach().requires_grad_(True)
            
            # Update control signal
            u_current = u_next
        
        # Final output
        self.y_hat = activations[-1]
        
        return activations, errors

    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass with EFC dynamic inversion.
        """
        # Run dynamic inversion
        activations, errors = self.dynamic_inversion(x, y)
        
        # Output is the last activation
        output = activations[-1]
        
        # Calculate loss for backward pass
        self.loss = self.compute_loss(output, y)
        
        # Register hooks for custom backward pass
        for i in range(1, len(activations)):
            # Need to create a closure to capture the correct error value
            def get_backward_hook(i, error):
                def backward_hook(grad):
                    # Replace gradient with teaching signal
                    return -error
                return backward_hook
            
            # Only register hook if tensor requires gradients (debug print to verify)
            if activations[i].requires_grad:
                activations[i].register_hook(get_backward_hook(i, errors[i-1]))
            else:
                print(f"Warning: Activation {i} does not require gradients")
        
        return output
    
    def backward(self):
        """
        Custom backward pass using hooks registered in forward_train.
        """
        # Standard backward pass will use our registered hooks
        self.loss.backward()
        
        # Clear the loss after backward to avoid accidental reuse
        del self.loss
    
    def calculate_fisher(self, dataloader):
        """Compute Fisher Information Matrix for continual learning"""
        fisher = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)
        
        # Set to evaluation mode
        self.eval()
        
        # Process each batch
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            outputs = self(inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # Handle different target formats
            if targets.dim() == 2:  # one-hot
                log_likelihood = (log_probs * targets).sum(dim=1)
            else:  # class indices
                log_likelihood = log_probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            # Compute gradients
            self.zero_grad()
            log_likelihood.sum().backward()
            
            # Accumulate squared gradients in Fisher
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2
        
        # Normalize by dataset size
        for n in fisher.keys():
            fisher[n] /= len(dataloader.dataset)
        
        return fisher
    
    def complete_task(self, dataloader):
        """Store Fisher and parameters after completing a task"""
        # Store current parameters
        self._means = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}
        
        # Calculate Fisher matrix
        current_fisher = self.calculate_fisher(dataloader)
        
        # Option to normalize Fisher (can be helpful for stability)
        if self.normalize_fisher:
            # Find maximum value for normalization
            max_fisher = max([tensor.max().item() for tensor in current_fisher.values()])
            if max_fisher > 0:
                for key in current_fisher:
                    current_fisher[key] = current_fisher[key] / max_fisher
        
        if self._first_task:
            self._fisher = current_fisher
            self._first_task = False
        else:
            # Accumulate Fisher matrices
            for n in self._fisher:
                if n in current_fisher:  # Safety check
                    self._fisher[n] += current_fisher[n]


class TaskIL_EFC_Network(TaskIL_DFC_Network):
    """
    Task-incremental learning version of EFC network.
    Maintains separate output heads for each task and applies Fisher modulation to shared parameters.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int], config: Any):
        """
        Initialize a task-incremental EFC network.
        
        Args:
            input_dim: Input dimension
            hidden_dims: List of hidden layer dimensions
            task_output_dims: List of output dimensions for each task
            config: Configuration object with EFC parameters
        """
        super().__init__(input_dim, hidden_dims, task_output_dims, config)
        
        # EFC-specific parameters
        self.beta = getattr(config, 'beta', 1.0)  # Fisher modulation strength
        
        # Use a slightly different activation function (softplus) which works better with multiplication
        self.activation = nn.Softplus()
        
        # Store task-specific Fisher matrices and means
        self.task_fisher = {}
        self.task_means = {}
        
    def activation_derivative(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute derivative of activation function.
        For softplus, it's sigmoid(x)
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Derivative of activation at input points
        """
        return torch.sigmoid(x)
    
    def compute_fisher_modulation(self, layer_idx: int) -> torch.Tensor:
        """
        Compute the Fisher-based modulation factor for a layer in task-incremental setting.
        
        Args:
            layer_idx: Index of the layer
            
        Returns:
            torch.Tensor: Fisher modulation factor for the layer
        """
        if self._first_task or self.current_task == 0:
            # No modulation for the first task
            return 0.0
            
        # Get layer parameters
        weight_name = f"layers.{layer_idx}.weight"
        bias_name = f"layers.{layer_idx}.bias"
        
        # Initialize gamma
        gamma = 0.0
        
        # Accumulate modulation from all previous tasks
        for task_id in range(self.current_task):
            if task_id not in self.task_fisher:
                continue
                
            # Get parameter difference from previous task
            weight_diff = self.layers[layer_idx].weight - self.task_means[task_id][weight_name]
            bias_diff = self.layers[layer_idx].bias - self.task_means[task_id][bias_name]
            
            # Get Fisher information
            weight_fisher = self.task_fisher[task_id][weight_name]
            bias_fisher = self.task_fisher[task_id][bias_name]
            
            # Add to modulation
            gamma -= self.beta * (weight_fisher * weight_diff).sum(dim=1) - self.beta * (bias_fisher * bias_diff)
            
        return gamma
    
    def dynamic_inversion_task_specific(self, x: torch.Tensor, y: torch.Tensor, task_id: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform dynamic inversion for task-specific output head with Fisher modulation.
        
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
        
        # Precompute Fisher modulation factors for shared layers
        gamma_factors = [self.compute_fisher_modulation(i) for i in range(len(self.layers))]
        
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
            
            # Compute Jacobian-vector products through the task-specific head
            # This requires careful handling since we have a shared feature extractor
            # and task-specific output heads
            
            # First, compute gradient through the output head
            features = activations[len(self.layers)]
            features.requires_grad_(True)
            output = self.output_heads[task_id](features)
            head_grad = torch.autograd.grad(
                output, features, grad_outputs=u, retain_graph=True, create_graph=False
            )[0]
            
            # Propagate this gradient backward through the shared layers
            psi_list = []
            current_grad = head_grad
            
            # Backward gradient propagation through shared layers
            for i in range(len(self.layers) - 1, -1, -1):
                input_act = activations[i]
                input_act.requires_grad_(True)
                output_act = self.layers[i](input_act)
                
                if i < len(self.layers) - 1:
                    output_act = self.activation(output_act)
                
                layer_grad = torch.autograd.grad(
                    output_act, input_act, grad_outputs=current_grad,
                    retain_graph=True, create_graph=False
                )[0]
                
                # Save psi value and update current gradient
                psi_list.insert(0, current_grad)
                current_grad = layer_grad
            
            # Update activations using control signals
            new_activations = [x]  # Input remains unchanged
            
            # Update shared layers with Fisher modulation
            for i in range(len(self.layers)):
                # Forward computation
                forward_input = new_activations[i]
                forward_output = self.layers[i](forward_input)
                
                # Apply activation function
                if i < len(self.layers) - 1:
                    forward_output = self.activation(forward_output)
                
                # Get layer-specific psi and gamma
                psi = psi_list[i]
                gamma = gamma_factors[i]
                
                # Compute modulated activation with the e^(psi+gamma) factor
                e_psi_gamma = torch.tanh(psi + gamma) + 1  # Ensures positive modulation
                modulated_output = e_psi_gamma * forward_output
                
                # Update activation with modulated value
                next_act = activations[i+1] + self.dt/self.tau_v * (modulated_output - activations[i+1])
                new_activations.append(next_act)
            
            # Update output layer (task-specific head)
            forward_output = self.output_heads[task_id](new_activations[-1])
            u_feedback = u  # Direct error signal for output layer
            next_act = activations[-1] + self.dt/self.tau_v * (forward_output - activations[-1] + u_feedback)
            new_activations.append(next_act)
            
            # Update activations for next iteration
            activations = new_activations
            
        return activations, errors
        
    def forward_train(self, x: torch.Tensor, y: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Training forward pass with Fisher-modulated dynamic inversion for task-specific head.
        
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
                    # For EFC, we replace the gradient with the error signal
                    return -error
                return backward_hook
            
            activations[i].register_hook(get_backward_hook(i, errors[i-1]))
            
        return output
    
    def complete_task(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
        """
        Complete a task and compute task-specific Fisher matrix.
        For EFC, this involves computing and storing the Fisher matrix for the current task.
        
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


class EFC_Network_CNN(nn.Module):
    """
    A convolutional implementation of EFC.
    This is a simplified implementation that focuses on the core ideas.
    """
    def __init__(self, input_channels: int, conv_channels: List[int], 
                hidden_dims: List[int], output_dim: int, config: Any):
        """
        Initialize a convolutional EFC network.
        
        Args:
            input_channels: Number of input channels
            conv_channels: List of convolutional layer channels
            hidden_dims: List of fully connected hidden layer dimensions
            output_dim: Output dimension
            config: Configuration object with EFC parameters
        """
        super().__init__()
        
        # Store network architecture
        self.input_channels = input_channels
        self.conv_channels = conv_channels
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        
        # EFC parameters
        self.beta = getattr(config, 'beta', 1.0)  # Fisher modulation strength
        self.dt = getattr(config, 'dt', 0.1)
        self.tau_v = getattr(config, 'tau_v', 0.1)
        self.k_p = getattr(config, 'k_p', 1.0)
        self.max_iter = getattr(config, 'max_iter', 50)
        self.tol = getattr(config, 'tol', 1e-4)
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = input_channels
        for out_channels in conv_channels:
            self.conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            in_channels = out_channels
        
        # Create pooling layer for transition to fully connected
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        fc_in_dim = conv_channels[-1]  # After pooling, each channel becomes a single value
        for fc_out_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(fc_in_dim, fc_out_dim))
            fc_in_dim = fc_out_dim
        
        # Output layer
        self.output_layer = nn.Linear(fc_in_dim, output_dim)
        
        # Activation function
        self.activation = nn.ReLU()  # ReLU is more stable for CNNs
        
        # Fisher information storage
        self._first_task = True
        self._fisher = {}
        self._means = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass for inference."""
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
        
        # Transition to fully connected
        x = self.pool(x).view(x.size(0), -1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = self.activation(fc_layer(x))
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def calculate_fisher(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix for CNN."""
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
    
    def complete_task(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> None:
        """Complete a task by computing and storing Fisher matrix."""
        self._means = {n: p.data.clone() for n, p in self.named_parameters()}
        
        if self._first_task:
            self._fisher = self.calculate_fisher(dataloader, device)
            self._first_task = False
        else:
            # Update Fisher with new task
            current_fisher = self.calculate_fisher(dataloader, device)
            for n in self._fisher:
                self._fisher[n] += current_fisher[n]
    
    def _calculate_psis_cnn(self, activations: List[torch.Tensor], control_signal: torch.Tensor) -> List[torch.Tensor]:
        """
        Calculate control signals (psis) for CNN layers.
        Uses autograd to compute gradients through the network.
        
        Args:
            activations: List of layer activations
            control_signal: Error feedback signal
            
        Returns:
            List[torch.Tensor]: Control signals for each layer
        """
        psi_list = []
        
        # We'll use autograd to compute gradients through the network
        # First, we need to recreate the forward pass with requires_grad=True
        with torch.enable_grad():
            # Create copies of activations that require gradients
            act_grads = []
            for act in activations:
                act_copy = act.detach().clone().requires_grad_(True)
                act_grads.append(act_copy)
            
            # Output layer computation
            output = self.output_layer(act_grads[-2])
            
            # Compute gradients with respect to control signal
            output_grad = torch.autograd.grad(
                outputs=output,
                inputs=act_grads,
                grad_outputs=control_signal,
                retain_graph=True
            )
            
            # Convert gradients to psi values
            psi_list = output_grad
            
        return psi_list
    
    def compute_fisher_modulation_cnn(self, layer_idx: int, is_conv: bool = True) -> torch.Tensor:
        """
        Compute the Fisher-based modulation factor for a CNN layer.
        
        Args:
            layer_idx: Index of the layer
            is_conv: Whether the layer is convolutional
            
        Returns:
            torch.Tensor: Fisher modulation factor
        """
        if self._first_task:
            # No modulation for the first task
            return 0.0
            
        # Get layer parameters
        if is_conv:
            layer = self.conv_layers[layer_idx]
            weight_name = f"conv_layers.{layer_idx}.weight"
            bias_name = f"conv_layers.{layer_idx}.bias"
        else:
            fc_idx = layer_idx - len(self.conv_layers)
            if fc_idx < len(self.fc_layers):
                layer = self.fc_layers[fc_idx]
                weight_name = f"fc_layers.{fc_idx}.weight"
                bias_name = f"fc_layers.{fc_idx}.bias"
            else:
                layer = self.output_layer
                weight_name = "output_layer.weight"
                bias_name = "output_layer.bias"
            
        # Get parameter difference from previous task
        weight_diff = layer.weight - self._means[weight_name]
        bias_diff = layer.bias - self._means[bias_name]
        
        # Get Fisher information
        weight_fisher = self._fisher[weight_name]
        bias_fisher = self._fisher[bias_name]
        
        # For convolutional layers, we need to handle the 4D tensor structure
        if is_conv:
            # Compute modulation (gamma term in the paper)
            # We sum over input channels and kernel dimensions
            gamma = -self.beta * (weight_fisher * weight_diff).sum(dim=(1, 2, 3))
            gamma = gamma - self.beta * (bias_fisher * bias_diff)
            
            # Reshape to match convolutional output
            return gamma.view(1, -1, 1, 1)
        else:
            # For fully connected layers, similar to the basic EFC
            gamma = -self.beta * (weight_fisher * weight_diff).sum(dim=1) - self.beta * (bias_fisher * bias_diff)
            return gamma
    
    def dynamic_inversion_cnn(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Perform dynamic inversion with Fisher modulation for CNN.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: Layer activations and errors
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Forward pass to initialize activations
        # Convolutional layers
        conv_acts = [x]
        conv_out = x
        for conv_layer in self.conv_layers:
            conv_out = conv_layer(conv_out)
            conv_out_activated = self.activation(conv_out)
            conv_acts.extend([conv_out, conv_out_activated])
        
        # Transition to fully connected
        fc_in = self.pool(conv_out_activated).view(batch_size, -1)
        
        # Fully connected layers
        fc_acts = [fc_in]
        fc_out = fc_in
        for fc_layer in self.fc_layers:
            fc_out = fc_layer(fc_out)
            fc_out_activated = self.activation(fc_out)
            fc_acts.extend([fc_out, fc_out_activated])
        
        # Output layer
        output = self.output_layer(fc_acts[-1])
        
        # Combine all activations (pre and post activation)
        all_acts = conv_acts + fc_acts + [output]
        
        # Initialize control signal and errors
        u = torch.zeros_like(output)
        errors = [torch.zeros_like(act) for act in all_acts[1:]]  # No error for input
        
        # Create a mask to track convergence
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Precompute Fisher modulation factors
        gamma_factors_conv = [self.compute_fisher_modulation_cnn(i, True) for i in range(len(self.conv_layers))]
        gamma_factors_fc = [self.compute_fisher_modulation_cnn(i, False) for i in range(len(self.fc_layers) + 1)]
        
        # Perform dynamic inversion
        for t in range(self.max_iter):
            # Compute prediction error
            pred_error = output - y
            
            # P controller for output layer
            u = self.k_p * pred_error
            
            # Check convergence
            converged = converged | (torch.norm(u, dim=1) < self.tol)
            if converged.all():
                break
            
            # Calculate control signals (psi values)
            psi_list = self._calculate_psis_cnn(all_acts, u)
            
            # Update convolutional activations with Fisher modulation
            new_conv_acts = [x]  # Input remains unchanged
            
            # Update conv layers
            for i in range(len(self.conv_layers)):
                # Get pre and post activation indices
                pre_idx = 2*i + 1
                post_idx = 2*i + 2
                
                # Get layer-specific psi and gamma
                psi = psi_list[pre_idx]
                gamma = gamma_factors_conv[i]
                
                # Compute modulated activation
                e_psi_gamma = torch.tanh(psi + gamma) + 1  # Ensures positive modulation
                modulated_output = e_psi_gamma * conv_acts[post_idx]
                
                # Update activation with modulated value
                conv_acts[post_idx] = conv_acts[post_idx] + self.dt/self.tau_v * (modulated_output - conv_acts[post_idx])
                
                # Add to new activations
                new_conv_acts.extend([conv_acts[pre_idx], conv_acts[post_idx]])
            
            # Transition to fully connected
            new_fc_in = self.pool(new_conv_acts[-1]).view(batch_size, -1)
            new_fc_acts = [new_fc_in]
            
            # Update FC layers
            for i in range(len(self.fc_layers)):
                # Get pre and post activation indices
                pre_idx = len(conv_acts) + 2*i + 1
                post_idx = len(conv_acts) + 2*i + 2
                
                # Get layer-specific psi and gamma
                psi = psi_list[pre_idx]
                gamma = gamma_factors_fc[i]
                
                # Compute modulated activation
                e_psi_gamma = torch.tanh(psi + gamma) + 1
                modulated_output = e_psi_gamma * fc_acts[2*i + 2]
                
                # Update activation
                fc_acts[2*i + 2] = fc_acts[2*i + 2] + self.dt/self.tau_v * (modulated_output - fc_acts[2*i + 2])
                
                # Add to new activations
                new_fc_acts.extend([fc_acts[2*i + 1], fc_acts[2*i + 2]])
            
            # Update output
            # Get output psi and gamma
            output_psi = psi_list[-1]
            output_gamma = gamma_factors_fc[-1]
            
            # Compute direct update for output (no activation function)
            output_new = self.output_layer(new_fc_acts[-1])
            e_psi_gamma = torch.tanh(output_psi + output_gamma) + 1
            modulated_output = output_new + u
            
            # Update output
            output = output + self.dt/self.tau_v * (modulated_output - output)
            
            # Update activations for next iteration
            conv_acts = new_conv_acts
            fc_acts = new_fc_acts
            
        # Combine final activations
        final_acts = conv_acts + fc_acts + [output]
        
        return final_acts, errors
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Training forward pass with Fisher-modulated dynamic inversion for CNN.
        
        Args:
            x: Input tensor
            y: Target tensor
            
        Returns:
            torch.Tensor: Output predictions
        """
        # Perform dynamic inversion to find steady-state activations and errors
        activations, errors = self.dynamic_inversion_cnn(x, y)
        
        # Extract output activations
        output = activations[-1]
        
        # Register hooks for custom backward pass
        for i in range(1, len(activations)):
            def get_backward_hook(i, error):
                def backward_hook(grad):
                    # For EFC, we replace the gradient with the error signal
                    return -error
                return backward_hook
            
            activations[i].register_hook(get_backward_hook(i, errors[i-1]))
            
        return output
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss (cross-entropy)."""
        return F.cross_entropy(output, target)


class TaskIL_EFC_Network_CNN(EFC_Network_CNN):
    """
    Task-incremental learning version of EFC CNN.
    Maintains separate output heads for each task.
    """
    def __init__(self, input_channels: int, conv_channels: List[int], 
                hidden_dims: List[int], task_output_dims: List[int], config: Any):
        """
        Initialize a task-incremental EFC CNN.
        
        Args:
            input_channels: Number of input channels
            conv_channels: List of convolutional layer channels
            hidden_dims: List of fully connected hidden layer dimensions
            task_output_dims: List of output dimensions for each task
            config: Configuration object with EFC parameters
        """
        # Initialize with shared feature extractor (minus the output layer)
        super().__init__(input_channels, conv_channels, hidden_dims, task_output_dims[0], config)
        
        # Remove the output layer
        delattr(self, 'output_layer')
        
        # Create task-specific output heads
        self.output_heads = nn.ModuleList()
        for task_dim in task_output_dims:
            self.output_heads.append(nn.Linear(hidden_dims[-1], task_dim))
            
        # Store task-specific Fisher matrices and means
        self.task_fisher = {}
        self.task_means = {}
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
            
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = self.activation(conv_layer(x))
        
        # Transition to fully connected
        x = self.pool(x).view(x.size(0), -1)
        
        # Fully connected layers
        for fc_layer in self.fc_layers:
            x = self.activation(fc_layer(x))
        
        # Task-specific output head
        return self.output_heads[task_id](x)
    
    def set_task(self, task_id: int) -> None:
        """Set the current task ID."""
        self.current_task = task_id
    
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