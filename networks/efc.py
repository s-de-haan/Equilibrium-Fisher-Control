import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from networks.base import EquilibriumModule


class EFCNetwork(nn.Module):
    """
    Corrected Autograd implementation of EFC
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
                 beta_softplus: float = 5.0):
        super().__init__()

        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])

        self.act = nn.Softplus(beta=beta_softplus)
        self.beta = beta

        # Dynamic inversion hyperparameters
        self.taus = taus if taus is not None else [dt] * len(self.layers)
        if len(self.taus) != len(self.layers):
            raise ValueError(f"len(taus): {len(self.taus)} must equal # layers: {len(self.layers)}")
            
        self.dt = dt
        self.k_p = k_p
        self.alpha = alpha
        self.tmax = tmax
        self.eps = eps

        # Fisher information storage
        self.fisher: dict[str, torch.Tensor] = {}
        self.theta_star: dict[str, torch.Tensor] = {}
        self.first_task = True

    def _forward_ff(self, x: torch.Tensor, requires_grad: bool = False) -> List[torch.Tensor]:
        """Forward pass with optional gradient requirements for intermediate activations"""
        acts = [x]
        
        for i, layer in enumerate(self.layers[:-1]):
            z = layer(acts[-1])
            if requires_grad:
                z.requires_grad_(True)
            r = self.act(z)
            if requires_grad:
                r.requires_grad_(True)
            acts.append(r)
            
        # Output layer (no activation)
        z_out = self.layers[-1](acts[-1])
        if requires_grad:
            z_out.requires_grad_(True)
        acts.append(z_out)
        
        return acts

    def _compute_gamma(self, layer_idx: int, batch_size: int, device: torch.device) -> torch.Tensor:
        """Compute Fisher preservation term for a specific layer"""
        if self.first_task:
            return torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)

        gamma = torch.zeros(batch_size, self.layers[layer_idx].out_features, device=device)
        layer = self.layers[layer_idx]
        
        # Weight contribution
        w_key = f"layers.{layer_idx}.weight"
        if w_key in self.fisher:
            w_diff = layer.weight - self.theta_star[w_key]
            w_fisher = self.fisher[w_key]
            # Sum over input dimension to get per-output-neuron gamma
            gamma += -self.beta * (w_fisher * w_diff).sum(dim=1).unsqueeze(0)
        
        # Bias contribution
        b_key = f"layers.{layer_idx}.bias"
        if b_key in self.fisher:
            b_diff = layer.bias - self.theta_star[b_key]
            b_fisher = self.fisher[b_key]
            gamma += -self.beta * (b_fisher * b_diff).unsqueeze(0)
            
        return gamma

    def _dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Dynamic inversion with proper gradient handling
        Returns: (equilibrium_activations, pseudo_loss)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize with feed-forward activations (with gradients)
        acts_ff = self._forward_ff(x, requires_grad=True)
        
        # Initialize state variables for hidden layers only (exclude input and output)
        num_hidden = len(self.layers) - 1  # Exclude output layer
        v_states = []  # Pre-activation states for hidden layers
        
        for i in range(num_hidden):
            layer = self.layers[i]
            v_ff = layer(acts_ff[i])
            v_states.append(v_ff.detach().clone().requires_grad_(True))
        
        # Control variables
        u = torch.zeros_like(targets, requires_grad=False)
        u_int = torch.zeros_like(targets, requires_grad=False)
        
        # Track current activations
        current_acts = [a.detach() for a in acts_ff]
        
        for iteration in range(self.tmax):
            # Rebuild computational graph for current iteration
            acts_with_grad = [x]  # Input (no gradients needed)
            
            # Hidden layers: use current v_states with activation
            for i in range(num_hidden):
                r_i = self.act(v_states[i])
                acts_with_grad.append(r_i)
            
            # Output layer
            y_current = self.layers[-1](acts_with_grad[-1])
            acts_with_grad.append(y_current)
            
            # PI controller
            error = y_current - targets
            u_int_next = u_int + self.dt * (error.detach() - self.alpha * u)
            u_next = u_int_next + self.k_p * error.detach()
            
            # Check convergence
            if torch.norm(u_next - u) < self.eps:
                break
            
            # Compute psi signals using autograd
            psis = []
            for i in range(len(acts_with_grad) - 2):  # Exclude input and output
                try:
                    psi = torch.autograd.grad(
                        outputs=y_current,
                        inputs=acts_with_grad[i + 1],  # Skip input
                        grad_outputs=u_next,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    if psi is None:
                        psi = torch.zeros_like(acts_with_grad[i + 1])
                    psis.append(psi.detach())
                except RuntimeError as e:
                    print(f"Gradient computation failed for layer {i}: {e}")
                    psis.append(torch.zeros_like(acts_with_grad[i + 1]))
            
            # Update v_states using Euler integration
            new_v_states = []
            for i in range(num_hidden):
                layer = self.layers[i]
                
                # Feed-forward pre-activation
                v_ff = layer(acts_with_grad[i])  # acts_with_grad[0] is input
                
                # Compute gamma (Fisher preservation)
                gamma = self._compute_gamma(i, batch_size, device)
                
                # Multiplicative modulation
                if i < len(psis):
                    psi_gamma = psis[i] + gamma
                    e_mod = torch.exp(torch.clamp(psi_gamma, -4, 4))  # Clamp for stability
                else:
                    e_mod = torch.ones_like(v_ff)
                
                # Euler update
                tau = self.taus[i]
                v_new = v_states[i] + tau * (v_ff.detach() * e_mod - v_states[i])
                v_new = v_new.detach().requires_grad_(True)
                new_v_states.append(v_new)
            
            # Update state
            v_states = new_v_states
            u_int, u = u_int_next.detach(), u_next.detach()
        
        # Compute final equilibrium activations (detached)
        final_acts = [x.detach()]
        for i in range(num_hidden):
            r_eq = self.act(v_states[i].detach())
            final_acts.append(r_eq)
        
        y_eq = self.layers[-1](final_acts[-1])
        final_acts.append(y_eq)
        
        # Compute pseudo-loss for learning
        pseudo_loss = 0.0
        acts_ff_clean = self._forward_ff(x, requires_grad=False)  # Clean forward pass
        
        for i in range(len(self.layers)):
            # Teaching signal = equilibrium - feedforward
            ts = final_acts[i + 1] - acts_ff_clean[i + 1]
            
            # Pre-activation with gradients for weights
            v_i = self.layers[i](acts_ff_clean[i])
            
            # Pseudo-loss accumulation
            pseudo_loss += (ts.detach() * v_i).sum()
        
        return final_acts, pseudo_loss

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None, use_efc: bool = True):
        """Forward pass - training or inference mode"""
        if self.training and targets is not None and use_efc:
            return self._forward_train(x, targets)
        else:
            # Standard inference
            return self._forward_ff(x, requires_grad=False)[-1]

    def _forward_train(self, x: torch.Tensor, targets: torch.Tensor):
        """Training forward pass using EFC"""
        # Convert targets to one-hot if needed
        if targets.dim() == 1:
            targets = F.one_hot(targets, self.layers[-1].out_features).float()

        # Dynamic inversion
        acts_eq, pseudo_loss = self._dynamic_inversion(x, targets)
        
        return acts_eq[-1], pseudo_loss
    
    def compute_fisher_information(self, dataloader):
        """Compute Fisher Information Matrix"""
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        
        self.eval()  # Set to eval mode
        num_samples = 0
        
        for x, y in dataloader:
            x = x.to(next(self.parameters()).device)
            y = y.to(next(self.parameters()).device)
            
            # Forward pass - ENABLE gradients for this computation
            logits = self(x, use_efc=False)  # Use standard forward for Fisher
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
            
            # Accumulate squared gradients (disable gradients for this part)
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        fisher[n] += p.grad.data.pow(2)
            
            num_samples += x.size(0)
        
        # Normalize by number of samples (disable gradients)
        with torch.no_grad():
            for n in fisher:
                fisher[n] /= num_samples
                
        return fisher

    def complete_task(self, dataloader):
        """Complete a task and update Fisher information"""
        # Store current parameters as optimal for this task
        self.theta_star = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}
        
        # Compute and store Fisher information
        if self.first_task:
            self.fisher = self.compute_fisher_information(dataloader)
            self.first_task = False
        else:
            new_fisher = self.compute_fisher_information(dataloader)
            # Accumulate Fisher information
            for n in self.fisher:
                if n in new_fisher:
                    self.fisher[n] += new_fisher[n]


class TaskILEFCNetwork(EFCNetwork):
    """Task-incremental variant of EFC"""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int],
                 task_output_dims: List[int],
                 **kwargs):
        # Initialize base with dummy output dimension
        super().__init__(input_dim, hidden_dims, hidden_dims[-1], **kwargs)
        
        # Remove the placeholder output layer
        self.layers = self.layers[:-1]
        
        # Add task-specific output heads
        self.output_heads = nn.ModuleList([
            nn.Linear(hidden_dims[-1], out_dim) for out_dim in task_output_dims
        ])
        
        self.current_task = 0

    def _forward_ff(self, x: torch.Tensor, task_id: Optional[int] = None, requires_grad: bool = False) -> List[torch.Tensor]:
        """Task-specific forward pass"""
        if task_id is None:
            task_id = self.current_task
            
        acts = [x]
        
        # Shared hidden layers
        for layer in self.layers:
            z = layer(acts[-1])
            if requires_grad:
                z.requires_grad_(True)
            r = self.act(z)
            if requires_grad:
                r.requires_grad_(True)
            acts.append(r)
        
        # Task-specific output
        z_out = self.output_heads[task_id](acts[-1])
        if requires_grad:
            z_out.requires_grad_(True)
        acts.append(z_out)
        
        return acts

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None, targets: Optional[torch.Tensor] = None, use_efc: bool = True):
        """Task-specific forward pass"""
        if task_id is None:
            task_id = self.current_task
            
        if self.training and targets is not None and use_efc:
            return self._forward_train(x, targets, task_id)
        else:
            return self._forward_ff(x, task_id, requires_grad=False)[-1]

    def _forward_train(self, x: torch.Tensor, targets: torch.Tensor, task_id: int):
        """Task-specific training forward pass"""
        if targets.dim() == 1:
            targets = F.one_hot(targets, self.output_heads[task_id].out_features).float()

        # Override the dynamic inversion to use task-specific head
        acts_eq, pseudo_loss = self._dynamic_inversion_task(x, targets, task_id)
        return acts_eq[-1], pseudo_loss

    def _dynamic_inversion_task(self, x: torch.Tensor, targets: torch.Tensor, task_id: int):
        """Task-specific dynamic inversion"""
        # Similar to parent but uses task-specific output head
        batch_size = x.shape[0]
        device = x.device
        
        # Feed-forward with task-specific head
        acts_ff = self._forward_ff(x, task_id, requires_grad=True)
        
        # Rest is similar to parent implementation...
        # (Implementation would be similar to parent _dynamic_inversion but using task_id)
        
        # For now, fall back to simpler implementation
        return self._dynamic_inversion_simple(x, targets, task_id)

    def _dynamic_inversion_simple(self, x: torch.Tensor, targets: torch.Tensor, task_id: int):
        """Simplified dynamic inversion for task-specific learning"""
        # Standard forward pass
        acts_ff = self._forward_ff(x, task_id, requires_grad=False)
        
        # For now, just return feed-forward result and compute simple pseudo-loss
        pseudo_loss = F.mse_loss(acts_ff[-1], targets)
        
        return acts_ff, pseudo_loss

    def set_task(self, task_id: int):
        """Set current task"""
        self.current_task = task_id


# Wrapper classes for compatibility with existing codebase
class EFC_Network_Wrapper(EquilibriumModule):
    """Wrapper for compatibility with existing training infrastructure"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, config, name="EFC_network"):
        super().__init__(input_dim, hidden_dims, output_dim)
        
        self.efc_network = EFCNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            beta=getattr(config, 'beta_efc', 1.0),
            taus=getattr(config, 'taus', None),
            tmax=getattr(config, 'tmax_di', 50),
            eps=getattr(config, 'eps', 1e-4),
            dt=getattr(config, 'dt_di', 0.1),
            k_p=getattr(config, 'k_p', 1.0)
        )
        
        self.name = name
        self.device = config.device
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        """Standard forward pass for inference"""
        return self.efc_network(x, use_efc=False)

    def forward_train(self, x, y):
        """Training forward with EFC"""
        if y.dim() == 2:
            y = y.argmax(dim=1)
        
        self.outputs, self.pseudo_loss = self.efc_network(x, targets=y, use_efc=True)
        
        return self.outputs

    def backward(self):
        """Backward pass handled by autograd"""
        pass

    def calculate_loss(self, y_hat, y):
        """Compute training loss"""
        if hasattr(self, "pseudo_loss"):
            # Log CE for monitoring
            if y.dim() == 2:
                y_labels = y.argmax(1)
            else:
                y_labels = y
            ce = self.loss_fn(y_hat.detach(), y_labels).item()
            self.last_ce = ce
            return self.pseudo_loss
        return self.loss_fn(y_hat, y)

    def complete_task(self, dataloader, device=None):
        """Complete task and update Fisher information"""
        self.efc_network.complete_task(dataloader)

    def to(self, device):
        """Move to device"""
        self.efc_network = self.efc_network.to(device)
        return super().to(device)


class TaskIL_EFC_Network_Wrapper(EFC_Network_Wrapper):
    """Task-incremental wrapper"""
    
    def __init__(self, config, num_tasks=5, task_output_size=2, name="TaskIL_EFC_network"):
        # Extract dimensions from config
        input_dim = config.layers[0]
        hidden_dims = config.layers[1:-1]
        task_output_dims = [task_output_size] * num_tasks
        
        # Initialize EquilibriumModule
        EquilibriumModule.__init__(self, input_dim, hidden_dims, task_output_size)
        
        # Create task-incremental EFC network
        self.efc_network = TaskILEFCNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            task_output_dims=task_output_dims,
            beta=getattr(config, 'beta_efc', 1.0),
            taus=getattr(config, 'taus', None),
            tmax=getattr(config, 'tmax_di', 50),
            eps=getattr(config, 'eps', 1e-4),
            dt=getattr(config, 'dt_di', 0.1),
            k_p=getattr(config, 'k_p', 1.0)
        )
        
        self.name = name
        self.device = config.device
        self.loss_fn = nn.CrossEntropyLoss()
        self.current_task = 0
        self.trained_tasks = set()

    def forward(self, x, task_id=None):
        """Task-specific forward pass"""
        if task_id is None:
            task_id = self.current_task
        return self.efc_network(x, task_id=task_id, use_efc=False)

    def forward_train(self, x, y):
        """Task-specific training forward pass"""
        if y.dim() == 2:
            y = y.argmax(dim=1)
        
        self.outputs, self.pseudo_loss = self.efc_network(
            x, task_id=self.current_task, targets=y, use_efc=True
        )
        return self.outputs

    def set_task(self, task_id):
        """Set current task"""
        self.current_task = task_id
        self.efc_network.set_task(task_id)

    def freeze_previous_tasks(self):
        """Placeholder for compatibility"""
        pass

    def complete_task_and_freeze_output_head(self, task_id):
        """Mark task as completed"""
        self.trained_tasks.add(task_id)