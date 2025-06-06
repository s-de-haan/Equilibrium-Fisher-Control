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
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 output_dim: int,
                 beta: float = 1.0,
                 taus: Optional[List[float]] = None,
                 dt: float = 8e-3,
                 k_p: float = 2.0,
                 alpha: float = 0.0,
                 tmax: int = 500,
                 eps: float = 1e-4,
                 beta_softplus: float = 5.0,
                 use_dynamic_inversion: bool = True):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim)
        
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])
        
        self.act = nn.Softplus(beta=beta_softplus)
        self.beta = beta
        self.use_dynamic_inversion = use_dynamic_inversion
        
        # Dynamic inversion hyperparameters
        self.taus = taus if taus is not None else [dt] * len(self.layers)
        self.dt = dt
        self.k_p = k_p
        self.alpha = alpha
        self.tmax = tmax
        self.eps = eps
        
        # Fisher information storage
        self.fisher = {}
        self.theta_star = {}
        self.first_task = True
        
        # For tracking modulation values (debugging)
        self._last_modulation = None
    
    def _compute_gamma(self, layer_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute Fisher preservation term γ = -β H_A (θ - θ*_A)"""
        if self.first_task:
            return torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)
        
        gamma = torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)
        layer = self.layers[layer_idx]
        
        # Weight contribution
        w_key = f"layers.{layer_idx}.weight"
        if w_key in self.fisher:
            w_diff = layer.weight - self.theta_star[w_key]
            w_fisher = self.fisher[w_key]
            gamma += -self.beta * (w_fisher * w_diff).sum(dim=1).unsqueeze(0)
        
        # Bias contribution
        b_key = f"layers.{layer_idx}.bias"
        if b_key in self.fisher:
            b_diff = layer.bias - self.theta_star[b_key]
            b_fisher = self.fisher[b_key]
            gamma += -self.beta * (b_fisher * b_diff).unsqueeze(0)
        
        return gamma
    
    def _forward_ff(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Standard feedforward pass"""
        acts = [x]
        for i, layer in enumerate(self.layers[:-1]):
            z = layer(acts[-1])
            r = self.act(z)
            acts.append(r)
        # Output layer (no activation)
        acts.append(self.layers[-1](acts[-1]))
        return acts
    
    def _non_dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Non-dynamical inversion (one-shot solution).
        Based on solving: min ||ψ||^2 s.t. e^(ψ+γ) f(φ,θ) - 1 = 0
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Get feedforward activations
        acts_ff = self._forward_ff(x)
        
        # Compute error at output
        error = targets - acts_ff[-1]
        
        # Compute effective Jacobian (using autograd for efficiency)
        J_eff = []
        for i in range(len(self.layers)):
            # Compute J_i^T u where u is the error
            jacob = torch.autograd.grad(
                outputs=acts_ff[-1],
                inputs=acts_ff[i],
                grad_outputs=error,
                retain_graph=True
            )[0]
            J_eff.append(jacob)
        
        # Compute optimal control signals
        equilibrium_acts = []
        pseudo_loss = 0.0
        
        for i in range(len(self.layers)):
            if i == 0:
                equilibrium_acts.append(x)
                continue
            
            # Compute ψ* + γ
            psi_star = J_eff[i] / (acts_ff[i] + 1e-8)  # Normalized by activation
            gamma = self._compute_gamma(i-1, batch_size, device)
            
            # CRITICAL: Use exponential, not tanh!
            modulation = torch.exp(psi_star + gamma)
            
            # Equilibrium activation
            r_eq = modulation * acts_ff[i]
            equilibrium_acts.append(r_eq)
            
            # Compute teaching signal
            ts = (r_eq - acts_ff[i]).detach()
            
            # Accumulate pseudo-loss
            v_i = self.layers[i-1](acts_ff[i-1])
            pseudo_loss += (ts * v_i).sum()
        
        # Output layer
        y_eq = self.layers[-1](equilibrium_acts[-1])
        equilibrium_acts.append(y_eq)
        
        # Final teaching signal
        ts_out = (y_eq - acts_ff[-1]).detach()
        v_out = self.layers[-1](acts_ff[-2])
        pseudo_loss += (ts_out * v_out).sum()
        
        return equilibrium_acts, pseudo_loss
    
    def _dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Dynamic inversion with proper exponential modulation.
        Solves the dynamics: τ ṙ = -r + e^(ψ+γ) φ(Wr_prev)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize with feedforward
        acts_ff = self._forward_ff(x)
        r_states = [a.clone().detach() for a in acts_ff]
        
        # Control variables
        u = torch.zeros_like(targets)
        u_int = torch.zeros_like(targets)
        
        # Track modulation for debugging
        self._last_modulation = []
        
        for iteration in range(self.tmax):
            # Error at output
            error = r_states[-1] - targets
            
            # PI controller
            u_int = u_int + self.dt * (error - self.alpha * u)
            u_next = u_int + self.k_p * error
            
            # Check convergence
            if torch.norm(u_next - u) < self.eps:
                break
            
            # Compute control signals via autograd
            psis = []
            for i in range(len(self.layers)):
                if i == 0:
                    psis.append(None)
                    continue
                
                # Jacobian-vector product: J_i^T u
                psi = torch.autograd.grad(
                    outputs=r_states[-1],
                    inputs=r_states[i],
                    grad_outputs=u_next,
                    retain_graph=True,
                    create_graph=True
                )[0]
                psis.append(psi)
            
            # Update states with exponential modulation
            new_r_states = [x]  # Input unchanged
            
            for i in range(len(self.layers)):
                if i == 0:
                    continue
                
                # Feedforward pre-activation
                v_ff = self.layers[i-1](new_r_states[i-1])
                r_ff = self.act(v_ff) if i < len(self.layers) else v_ff
                
                # Compute modulation
                psi = psis[i] / (r_ff + 1e-8)  # Normalize by activation
                gamma = self._compute_gamma(i-1, batch_size, device)
                
                # CRITICAL: Exponential modulation!
                modulation = torch.exp(psi + gamma)
                self._last_modulation.append(modulation.detach())
                
                # Update with dynamics
                tau = self.taus[i-1]
                r_new = r_states[i] + tau * (modulation * r_ff - r_states[i])
                new_r_states.append(r_new)
            
            # Output layer
            y_new = self.layers[-1](new_r_states[-1])
            new_r_states.append(y_new)
            
            r_states = [r.detach() for r in new_r_states]
            u = u_next
        
        # Compute pseudo-loss
        pseudo_loss = 0.0
        for i in range(len(self.layers)):
            ts = (r_states[i+1] - acts_ff[i+1]).detach()
            v_i = self.layers[i](acts_ff[i])
            
            # Include modulation in the loss
            if i > 0:
                mod = self._last_modulation[i-1]
                # Weight update proportional to (e^(ψ+γ) - 1)
                weight_factor = (mod - 1).mean()
                pseudo_loss += weight_factor * (ts * v_i).sum()
            else:
                pseudo_loss += (ts * v_i).sum()
        
        return r_states, pseudo_loss
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, use_efc: bool = True):
        """Forward pass - training or inference mode"""
        if self.training and targets is not None and use_efc:
            return self._forward_train(x, targets)
        else:
            # Standard inference
            return self._forward_ff(x)[-1]
    
    def _forward_train(self, x: torch.Tensor, targets: torch.Tensor):
        """Training forward pass using EFC"""
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, self.layers[-1].out_features).float()
        
        # Choose inversion method
        if self.use_dynamic_inversion:
            acts_eq, pseudo_loss = self._dynamic_inversion(x, targets)
        else:
            acts_eq, pseudo_loss = self._non_dynamic_inversion(x, targets)
        
        return acts_eq[-1], pseudo_loss
    
    def compute_fisher_information(self, dataloader):
        """Compute Fisher Information Matrix"""
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        
        self.eval()
        num_samples = 0
        
        for x, y in dataloader:
            x = x.to(next(self.parameters()).device)
            y = y.to(next(self.parameters()).device)
            
            # Forward pass
            logits = self(x, use_efc=False)
            log_probs = F.log_softmax(logits, dim=1)
            
            # Convert y to one-hot if needed
            if y.dim() == 1:
                y_onehot = F.one_hot(y, logits.size(1)).float()
            else:
                y_onehot = y
            
            # Log-likelihood
            log_likelihood = (log_probs * y_onehot).sum()
            
            # Compute gradients
            self.zero_grad()
            log_likelihood.backward()
            
            # Accumulate squared gradients
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data.pow(2)
            
            num_samples += x.size(0)
        
        # Normalize
        with torch.no_grad():
            for n in fisher:
                fisher[n] /= num_samples
        
        return fisher
    
    def complete_task(self, dataloader):
        """Complete a task and update Fisher information"""
        self.theta_star = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}
        
        if self.first_task:
            self.fisher = self.compute_fisher_information(dataloader)
            self.first_task = False
        else:
            new_fisher = self.compute_fisher_information(dataloader)
            for n in self.fisher:
                if n in new_fisher:
                    self.fisher[n] += new_fisher[n]


class TaskILEFCNetwork(BaseTaskIncrementalNetwork):
    """
    Task-Incremental Equilibrium Fisher Control (EFC) Network.
    
    Architecture: shared feature trunk + task-specific heads
    The trunk is trained with EFC regularized by Fisher information of previous tasks.
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 task_output_dims: List[int],
                 beta: float = 1.0,
                 taus: Optional[List[float]] = None,
                 dt: float = 8e-3,
                 k_p: float = 2.0,
                 alpha: float = 0.0,
                 tmax: int = 500,
                 eps: float = 1e-4,
                 beta_softplus: float = 5.0,
                 use_dynamic_inversion: bool = True):
        super().__init__(input_dim, hidden_dims, task_output_dims)

        # Remove the output layer from base network (it adds one automatically)
        # We only want the shared trunk
        self.layers = self.layers[:-1] if len(self.layers) > len(hidden_dims) else self.layers

        # Recreate shared trunk layers properly
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])

        # Task-specific output heads (inherited from base class)
        # self.output_heads is already created by BaseTaskIncrementalNetwork

        # Activation function
        self.act = nn.Softplus(beta=beta_softplus)

        # EFC hyperparameters
        self.beta = beta
        self.taus = taus if taus is not None else [dt] * len(self.layers)
        self.dt = dt
        self.k_p = k_p
        self.alpha = alpha
        self.tmax = tmax
        self.eps = eps
        self.use_dynamic_inversion = use_dynamic_inversion

        # Fisher information storage
        self.fisher = {}
        self.theta_star = {}
        self.first_task = True

        # Debug storage
        self._last_modulation = None
        self._pseudo_loss = None

    def _compute_gamma(self, layer_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute Fisher preservation term γ = -β H_A (θ - θ*_A) for shared layers only."""
        if self.first_task:
            return torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)

        gamma = torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)
        layer = self.layers[layer_idx]

        # Weight contribution
        w_key = f"layers.{layer_idx}.weight"
        if w_key in self.fisher:
            w_diff = layer.weight - self.theta_star[w_key]
            w_fisher = self.fisher[w_key]
            gamma += -self.beta * (w_fisher * w_diff).sum(dim=1).unsqueeze(0)

        # Bias contribution
        b_key = f"layers.{layer_idx}.bias"
        if b_key in self.fisher:
            b_diff = layer.bias - self.theta_star[b_key]
            b_fisher = self.fisher[b_key]
            gamma += -self.beta * (b_fisher * b_diff).unsqueeze(0)

        return gamma

    def _task_feedforward(self, x: torch.Tensor, task_id: int) -> List[torch.Tensor]:
        """Feedforward pass through shared trunk + task head."""
        acts = [x]
        
        # Pass through shared layers
        for layer in self.layers:
            x = self.act(layer(x))
            acts.append(x)
        
        # Pass through task-specific head
        y = self.output_heads[task_id](x)
        acts.append(y)
        
        return acts

    def _task_non_dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor, task_id: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Non-dynamic inversion for task-incremental setting."""
        batch_size = x.shape[0]
        device = x.device
        
        # Get feedforward activations
        acts_ff = self._task_feedforward(x, task_id)
        
        # Error at output
        error = targets - acts_ff[-1]
        
        # Compute Jacobian-vector products for shared layers only
        J_eff = [None]  # No gradient for input
        for i in range(1, len(self.layers) + 1):  # For each shared layer
            if acts_ff[i].requires_grad or True:  # Ensure gradients are enabled
                acts_ff[i].requires_grad_(True)
            
            jac = torch.autograd.grad(
                outputs=acts_ff[-1],
                inputs=acts_ff[i],
                grad_outputs=error,
                retain_graph=True,
                create_graph=True
            )[0]
            J_eff.append(jac)

        # Compute equilibrium activations
        equilibrium_acts = [x]  # Input unchanged
        pseudo_loss = torch.tensor(0.0, device=device)

        # Update shared layer activations
        for i in range(1, len(self.layers) + 1):
            # Compute control signal
            psi = J_eff[i] / (acts_ff[i] + 1e-8)
            gamma = self._compute_gamma(i-1, batch_size, device)
            
            # Exponential modulation
            modulation = torch.exp(psi + gamma)
            r_eq = modulation * acts_ff[i]
            equilibrium_acts.append(r_eq)
            
            # Teaching signal for weight updates
            teaching_signal = (r_eq - acts_ff[i]).detach()
            pre_activation = self.layers[i-1](acts_ff[i-1])
            pseudo_loss += (teaching_signal * pre_activation).sum()

        # Output layer (task head)
        y_eq = self.output_heads[task_id](equilibrium_acts[-1])
        equilibrium_acts.append(y_eq)
        
        # Teaching signal for output head
        ts_out = (y_eq - acts_ff[-1]).detach()
        pre_act_out = equilibrium_acts[-2]  # Features from last shared layer
        pseudo_loss += (ts_out * pre_act_out).sum()

        return equilibrium_acts, pseudo_loss

    def _calculate_task_psis(self, u: torch.Tensor, task_id: int) -> List[torch.Tensor]:
        """Calculate control signals (psis) for each layer - following original _calculate_psis pattern."""
        L = len(self.layers)
        psi_list = [None] * L
        
        # Start with the output error
        psi = u  # Shape: (batch_size, 2)
        
        # Backward from last shared layer to first
        for i in range(L - 1, -1, -1):
            if i == L - 1:
                # From task head back to last shared layer
                # (256, 2) @ (2, 400) = (256, 400)
                psi = psi @ self.output_heads[task_id].weight
            else:
                # Between shared layers  
                # (256, 400) @ (400, 400) = (256, 400)
                psi = psi @ self.layers[i + 1].weight
            
            # Apply activation derivative (simplified)
            # Just use the pattern from original EFC
            psi_list[i] = psi
        
        return psi_list


    def _task_dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor, task_id: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Dynamic inversion following the exact pattern of EFCNetwork._dynamic_inversion."""
        batch_size = x.shape[0]
        device = x.device
        
        # Store input for psi calculation
        self.input = x
        
        # Initialize with feedforward
        acts_ff = self._task_feedforward(x, task_id)
        r_states = [a.clone().detach() for a in acts_ff]
        
        # Control variables
        u = torch.zeros_like(targets)
        u_int = torch.zeros_like(targets)
        
        self._last_modulation = []
        
        for iteration in range(self.tmax):
            # Error at output
            error = r_states[-1] - targets
            
            # PI controller
            u_int = u_int + self.dt * (error - self.alpha * u)
            u_next = u_int + self.k_p * error
            
            # Check convergence
            if torch.norm(u_next - u) < self.eps:
                break
            
            # Calculate psis using the efficient method
            psis = self._calculate_task_psis(u_next, task_id)
            
            # Update states with exponential modulation
            new_r_states = [x]  # Input unchanged
            
            for i in range(len(self.layers)):
                # Feedforward pre-activation
                pre_activation = self.layers[i](new_r_states[i])
                r_ff = self.act(pre_activation)
                
                # Control signal with Fisher preservation
                psi = psis[i]  # psi for this layer (now correctly indexed)
                gamma = self._compute_gamma(i, batch_size, device)
                
                # Exponential modulation
                modulation = torch.exp(psi + gamma)
                self._last_modulation.append(modulation.detach())
                
                # Update dynamics
                tau = self.taus[i]
                r_new = r_states[i+1] + tau * (modulation * r_ff - r_states[i+1])
                new_r_states.append(r_new)
            
            # Update output (task head)
            y_new = self.output_heads[task_id](new_r_states[-1])
            new_r_states.append(y_new)
            
            # Detach for next iteration
            r_states = [r.detach() for r in new_r_states]
            u = u_next
        
        # Compute pseudo-loss
        pseudo_loss = torch.tensor(0.0, device=device)
        
        for i in range(len(self.layers)):
            # Teaching signal
            ts = (r_states[i+1] - acts_ff[i+1]).detach()
            v_i = self.layers[i](acts_ff[i])
            
            # Include modulation in the loss
            if self._last_modulation and len(self._last_modulation) > i:
                mod = self._last_modulation[i]
                weight_factor = (mod - 1).mean()
                pseudo_loss += weight_factor * (ts * v_i).sum()
            else:
                pseudo_loss += (ts * v_i).sum()
        
        # Output layer teaching signal
        ts_out = (r_states[-1] - acts_ff[-1]).detach()
        v_out = self.output_heads[task_id](acts_ff[-2])
        pseudo_loss += (ts_out * v_out).sum()
        
        return r_states, pseudo_loss

    def forward_train(self, x: torch.Tensor, y: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Training forward pass with EFC - OVERRIDE BASE CLASS."""
        if task_id is None:
            task_id = self.current_task
        
        # Convert targets to one-hot if needed
        if y.dim() == 1:
            y = F.one_hot(y, self.output_heads[task_id].out_features).float()
        
        # Choose inversion method
        if self.use_dynamic_inversion:
            acts_eq, pseudo_loss = self._task_dynamic_inversion(x, y, task_id)
        else:
            acts_eq, pseudo_loss = self._task_non_dynamic_inversion(x, y, task_id)
        
        # Store pseudo-loss for use in compute_loss
        self._pseudo_loss = pseudo_loss
        
        return acts_eq[-1]

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """Standard forward pass for inference."""
        if task_id is None:
            task_id = self.current_task
        
        # Standard feedforward (no EFC)
        return self._task_feedforward(x, task_id)[-1]

    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss including EFC pseudo-loss."""
        # Standard cross-entropy loss
        if target.dim() == 2 and target.shape[1] > 1:
            target = target.argmax(dim=1)
        task_loss = F.cross_entropy(output, target)
        
        # Add pseudo-loss if available
        if hasattr(self, '_pseudo_loss') and self._pseudo_loss is not None:
            total_loss = self._pseudo_loss
            # Reset pseudo-loss
            self._pseudo_loss = None
            return total_loss
        
        return task_loss

    def _compute_fisher_information(self, dataloader):
        """Compute Fisher Information Matrix for shared parameters only."""
        fisher = {}
        # Only compute Fisher for shared trunk parameters
        for n, p in self.named_parameters():
            if p.requires_grad and 'layers.' in n:  # Only shared layers
                fisher[n] = torch.zeros_like(p)
        
        self.eval()
        num_samples = 0
        device = next(self.parameters()).device
        
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass (inference mode)
            outputs = self.forward(inputs, task_id=self.current_task)
            log_probs = F.log_softmax(outputs, dim=1)
            
            # Convert targets to proper format
            if targets.dim() == 2:  # one-hot
                log_likelihood = (log_probs * targets).sum(dim=1)
            else:  # class indices
                log_likelihood = log_probs.gather(1, targets.unsqueeze(1)).squeeze()
            
            # Compute gradients
            self.zero_grad()
            log_likelihood.sum().backward()
            
            # Accumulate squared gradients (shared layers only)
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None and 'layers.' in n:
                    fisher[n] += p.grad.data ** 2
            
            num_samples += inputs.size(0)
        
        # Normalize by dataset size
        for n in fisher:
            fisher[n] /= num_samples
        
        return fisher

    def complete_task(self, dataloader):
        """Complete a task and update Fisher information."""
        # Store current parameters (shared layers only)
        self.theta_star = {
            n: p.data.clone() 
            for n, p in self.named_parameters() 
            if p.requires_grad and 'layers.' in n
        }
        
        if self.first_task:
            self.fisher = self._compute_fisher_information(dataloader)
            self.first_task = False
        else:
            new_fisher = self._compute_fisher_information(dataloader)
            for n in self.fisher:
                if n in new_fisher:
                    self.fisher[n] += new_fisher[n]