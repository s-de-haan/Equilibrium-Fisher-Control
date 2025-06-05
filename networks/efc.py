import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from networks.base import BaseNetwork, BaseTaskIncrementalNetwork


class EFCNetwork(BaseNetwork):
    """
    Equilibrium Fisher Control (EFC) Network.
    Clean implementation using autograd and unified base interface.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 beta: float = 1.0, dt: float = 0.008, k_p: float = 2.0,
                 alpha: float = 1e-4, max_iter: int = 500, eps: float = 1e-4,
                 beta_softplus: float = 5.0):
        super().__init__(input_dim, hidden_dims, output_dim)
        
        # Use softplus activation for EFC
        self.activation = nn.Softplus(beta=beta_softplus)
        
        # EFC parameters
        self.beta = beta  # Fisher preservation strength
        self.dt = dt
        self.k_p = k_p
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """EFC training forward pass with dynamic inversion."""
        # Convert targets to one-hot if needed
        if y.dim() == 1:
            y_onehot = F.one_hot(y, self.output_dim).float()
        else:
            y_onehot = y
        
        # Perform dynamic inversion
        output, _ = self._dynamic_inversion(x, y_onehot)
        return output
    
    def _dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Dynamic inversion with Fisher preservation using autograd.
        Returns (equilibrium_output, pseudo_loss)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize with feedforward activations
        with torch.no_grad():
            acts_ff = self._feedforward_pass(x)
        
        # Initialize states for hidden layers (requires gradients for autograd)
        states = []
        for i in range(len(self.layers) - 1):  # Exclude output layer
            state = acts_ff[i + 1].detach().clone()
            state.requires_grad_(True)
            states.append(state)
        
        # Control variables
        u = torch.zeros_like(targets)
        u_int = torch.zeros_like(targets)
        
        # Dynamic inversion loop
        for iteration in range(self.max_iter):
            # Build computational graph for current iteration
            current_acts = [x]  # Input (no gradient needed)
            
            # Hidden layers: apply activation to states
            for i in range(len(states)):
                activated = self.activation(states[i])
                current_acts.append(activated)
            
            # Output layer
            output = self.layers[-1](current_acts[-1])
            current_acts.append(output)
            
            # PI controller
            error = output - targets
            u_int_next = u_int + self.dt * (error.detach() - self.alpha * u)
            u_next = u_int_next + self.k_p * error.detach()
            
            # Check convergence
            if torch.norm(u_next - u) < self.eps:
                break
            
            # Compute control signals using autograd
            psis = self._compute_psis_autograd(output, current_acts[1:-1], u_next)
            
            # Update states using Euler integration
            new_states = []
            for i in range(len(states)):
                # Feedforward pre-activation
                v_ff = self.layers[i](current_acts[i])
                
                # Fisher preservation term
                gamma = self._compute_gamma(i, batch_size, device)
                
                # Multiplicative modulation
                psi_gamma = psis[i] + gamma
                e_mod = torch.exp(torch.clamp(psi_gamma, -3, 3))  # Clamp for stability
                
                # Euler update
                new_state = states[i] + self.dt * (v_ff.detach() * e_mod - states[i])
                new_state = new_state.detach()
                new_state.requires_grad_(True)
                new_states.append(new_state)
            
            # Update control and states
            states = new_states
            u_int, u = u_int_next.detach(), u_next.detach()
        
        # Final forward pass for output
        final_acts = [x]
        for i, state in enumerate(states):
            final_acts.append(self.activation(state))
        final_output = self.layers[-1](final_acts[-1])
        
        # Compute pseudo-loss for gradient-based learning
        pseudo_loss = self._compute_pseudo_loss(x, final_output, targets)
        
        return final_output, pseudo_loss
    
    def _feedforward_pass(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Standard feedforward pass."""
        activations = [x]
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            activations.append(x)
        x = self.layers[-1](x)
        activations.append(x)
        return activations
    
    def _compute_psis_autograd(self, output: torch.Tensor, hidden_acts: List[torch.Tensor], 
                              u: torch.Tensor) -> List[torch.Tensor]:
        """Compute control signals using autograd."""
        psis = []
        
        for i, act in enumerate(hidden_acts):
            try:
                # Compute gradient of output w.r.t. hidden activation
                psi = torch.autograd.grad(
                    outputs=output,
                    inputs=act,
                    grad_outputs=u,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if psi is None:
                    psi = torch.zeros_like(act)
                
                psis.append(psi.detach())
                
            except RuntimeError:
                # Fallback to zeros if gradient computation fails
                psis.append(torch.zeros_like(act))
        
        return psis
    
    def _compute_gamma(self, layer_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute Fisher preservation term."""
        if self._first_task:
            return torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)
        
        gamma = torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)
        layer = self.layers[layer_idx]
        
        # Weight contribution
        w_key = f"layers.{layer_idx}.weight"
        if w_key in self._fisher:
            w_diff = layer.weight - self._means[w_key]
            w_fisher = self._fisher[w_key]
            gamma += -self.beta * (w_fisher * w_diff).sum(dim=1).unsqueeze(0)
        
        # Bias contribution
        b_key = f"layers.{layer_idx}.bias"
        if b_key in self._fisher:
            b_diff = layer.bias - self._means[b_key]
            b_fisher = self._fisher[b_key]
            gamma += -self.beta * (b_fisher * b_diff).unsqueeze(0)
        
        return gamma
    
    def _compute_pseudo_loss(self, x: torch.Tensor, equilibrium_output: torch.Tensor, 
                           targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pseudo-loss for gradient-based parameter updates.
        This creates the teaching signals that drive learning.
        """
        # Get feedforward activations
        ff_acts = self._feedforward_pass(x)
        
        # Teaching signal is difference between equilibrium and feedforward
        teaching_signal = equilibrium_output - ff_acts[-1]
        
        # Create pseudo-loss that will generate the right gradients
        pseudo_loss = -(teaching_signal.detach() * ff_acts[-1]).sum()
        
        return pseudo_loss
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute loss. For EFC, we use the pseudo-loss if available,
        otherwise fall back to standard cross-entropy.
        """
        if hasattr(self, '_pseudo_loss'):
            return self._pseudo_loss
        else:
            return super().compute_loss(output, target)


class TaskILEFCNetwork(BaseTaskIncrementalNetwork):
    """Task-incremental EFC network."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], task_output_dims: List[int],
                 beta: float = 1.0, dt: float = 0.008, k_p: float = 2.0,
                 alpha: float = 1e-4, max_iter: int = 500, eps: float = 1e-4,
                 beta_softplus: float = 5.0):
        super().__init__(input_dim, hidden_dims, task_output_dims)
        
        # Use softplus activation
        self.activation = nn.Softplus(beta=beta_softplus)
        
        # EFC parameters
        self.beta = beta
        self.dt = dt
        self.k_p = k_p
        self.alpha = alpha
        self.max_iter = max_iter
        self.eps = eps
    
    def forward_train(self, x: torch.Tensor, y: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Task-specific EFC training forward pass."""
        if task_id is None:
            task_id = self.current_task
        
        # Convert targets to one-hot
        if y.dim() == 1:
            output_dim = self.output_heads[task_id].out_features
            y_onehot = F.one_hot(y, output_dim).float()
        else:
            y_onehot = y
        
        # Perform task-specific dynamic inversion
        output, pseudo_loss = self._task_dynamic_inversion(x, y_onehot, task_id)
        self._pseudo_loss = pseudo_loss
        
        return output
    
    def _task_dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor, 
                               task_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Task-specific dynamic inversion."""
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize with feedforward pass
        with torch.no_grad():
            acts_ff = self._task_feedforward_pass(x, task_id)
        
        # Initialize states for shared layers
        states = []
        for i in range(len(self.layers)):
            state = acts_ff[i + 1].detach().clone()
            state.requires_grad_(True)
            states.append(state)
        
        # Control variables
        u = torch.zeros_like(targets)
        u_int = torch.zeros_like(targets)
        
        # Dynamic inversion loop
        for iteration in range(self.max_iter):
            # Build computational graph
            current_acts = [x]
            
            # Shared layers
            for i, state in enumerate(states):
                activated = self.activation(state)
                current_acts.append(activated)
            
            # Task-specific output
            output = self.output_heads[task_id](current_acts[-1])
            
            # PI controller
            error = output - targets
            u_int_next = u_int + self.dt * (error.detach() - self.alpha * u)
            u_next = u_int_next + self.k_p * error.detach()
            
            # Check convergence
            if torch.norm(u_next - u) < self.eps:
                break
            
            # Compute control signals
            psis = self._compute_task_psis_autograd(output, current_acts[1:], u_next)
            
            # Update states
            new_states = []
            for i in range(len(states)):
                v_ff = self.layers[i](current_acts[i])
                gamma = self._compute_gamma(i, batch_size, device)
                
                psi_gamma = psis[i] + gamma
                e_mod = torch.exp(torch.clamp(psi_gamma, -3, 3))
                
                new_state = states[i] + self.dt * (v_ff.detach() * e_mod - states[i])
                new_state = new_state.detach()
                new_state.requires_grad_(True)
                new_states.append(new_state)
            
            states = new_states
            u_int, u = u_int_next.detach(), u_next.detach()
        
        # Final forward pass
        final_acts = [x]
        for state in states:
            final_acts.append(self.activation(state))
        final_output = self.output_heads[task_id](final_acts[-1])
        
        # Compute pseudo-loss
        pseudo_loss = self._compute_task_pseudo_loss(x, final_output, targets, task_id)
        
        return final_output, pseudo_loss
    
    def _task_feedforward_pass(self, x: torch.Tensor, task_id: int) -> List[torch.Tensor]:
        """Task-specific feedforward pass."""
        activations = [x]
        for layer in self.layers:
            x = self.activation(layer(x))
            activations.append(x)
        x = self.output_heads[task_id](x)
        activations.append(x)
        return activations
    
    def _compute_task_psis_autograd(self, output: torch.Tensor, hidden_acts: List[torch.Tensor],
                                   u: torch.Tensor) -> List[torch.Tensor]:
        """Compute task-specific control signals."""
        psis = []
        
        for act in hidden_acts:
            try:
                psi = torch.autograd.grad(
                    outputs=output,
                    inputs=act,
                    grad_outputs=u,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if psi is None:
                    psi = torch.zeros_like(act)
                
                psis.append(psi.detach())
                
            except RuntimeError:
                psis.append(torch.zeros_like(act))
        
        return psis
    
    def _compute_gamma(self, layer_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute Fisher preservation term for shared layers."""
        if self._first_task:
            return torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)
        
        gamma = torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)
        layer = self.layers[layer_idx]
        
        # Use task-specific Fisher if available, otherwise use accumulated Fisher
        if self.current_task in self.task_fisher:
            fisher = self.task_fisher[self.current_task]
            means = self.task_means[self.current_task]
        else:
            fisher = self._fisher
            means = self._means
        
        # Weight contribution
        w_key = f"layers.{layer_idx}.weight"
        if w_key in fisher:
            w_diff = layer.weight - means[w_key]
            w_fisher = fisher[w_key]
            gamma += -self.beta * (w_fisher * w_diff).sum(dim=1).unsqueeze(0)
        
        # Bias contribution
        b_key = f"layers.{layer_idx}.bias"
        if b_key in fisher:
            b_diff = layer.bias - means[b_key]
            b_fisher = fisher[b_key]
            gamma += -self.beta * (b_fisher * b_diff).unsqueeze(0)
        
        return gamma
    
    def _compute_task_pseudo_loss(self, x: torch.Tensor, equilibrium_output: torch.Tensor,
                                 targets: torch.Tensor, task_id: int) -> torch.Tensor:
        """Compute task-specific pseudo-loss."""
        ff_acts = self._task_feedforward_pass(x, task_id)
        teaching_signal = equilibrium_output - ff_acts[-1]
        pseudo_loss = -(teaching_signal.detach() * ff_acts[-1]).sum()
        return pseudo_loss
    
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Task-incremental loss computation."""
        if hasattr(self, '_pseudo_loss'):
            return self._pseudo_loss
        else:
            return super().compute_loss(output, target)