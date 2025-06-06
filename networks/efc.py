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
    """Equilibrium‑Fisher‑Control network for *task‑incremental* learning.

    The architecture is a shared feature trunk trained with EFC
    regularised by Fisher information of *previous tasks*, plus one
    linear (or low‑capacity) head per task.  During training we can call ▸

        logits, pseudo_loss = model.forward(x, y, task_id)

    and add the returned *pseudo_loss* to the usual cross‑entropy (or
    use the convenience `compute_loss`).
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

        # ---------- trunk (shared) ----------
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        ])

        # ---------- heads (task‑specific) ----------
        head_in = hidden_dims[-1]
        self.output_heads = nn.ModuleList([
            nn.Linear(head_in, out_d) for out_d in task_output_dims
        ])

        # ---------- nonlinearities ----------
        self.act = nn.Softplus(beta=beta_softplus)

        # ---------- EFC hyper‑parameters ----------
        self.beta = beta
        self.taus = taus if taus is not None else [dt] * len(self.layers)
        self.dt = dt
        self.k_p = k_p
        self.alpha = alpha
        self.tmax = tmax
        self.eps = eps
        self.use_dynamic_inversion = use_dynamic_inversion

        # ---------- Fisher bookkeeping ----------
        self.fisher = {}
        self.theta_star = {}
        self.first_task = True

        # debug storage
        self._last_modulation = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _compute_gamma(self, layer_idx: int, B: int, device: torch.device) -> torch.Tensor:
        """γ = −β H (θ − θ★)  for the *shared* layer `layer_idx`."""
        if self.first_task:
            return torch.zeros(B, self.layers[layer_idx].out_features, device=device)

        gamma = torch.zeros(B, self.layers[layer_idx].out_features, device=device)
        layer = self.layers[layer_idx]

        # weight contribution
        w_key = f"layers.{layer_idx}.weight"
        if w_key in self.fisher:
            diff = layer.weight - self.theta_star[w_key]
            gamma += -self.beta * (self.fisher[w_key] * diff).sum(dim=1).unsqueeze(0)

        # bias contribution
        b_key = f"layers.{layer_idx}.bias"
        if b_key in self.fisher:
            diff = layer.bias - self.theta_star[b_key]
            gamma += -self.beta * (self.fisher[b_key] * diff).unsqueeze(0)

        return gamma

    # ------------------------------------------------------------------
    # feed‑forward utilities
    # ------------------------------------------------------------------
    def _task_feedforward(self, x: torch.Tensor, task_id: int) -> List[torch.Tensor]:
        """Return activations `[x, r¹, …, rᴸ, y]` for the given task head."""
        acts = [x]
        for layer in self.layers:
            x = self.act(layer(x))
            acts.append(x)
        # final task‑specific linear
        y = self.output_heads[task_id](x)
        acts.append(y)
        return acts

    # ------------------------------------------------------------------
    # non‑dynamic inversion (analytic, one‑shot)
    # ------------------------------------------------------------------
    def _task_non_dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor, task_id: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B, device = x.shape[0], x.device
        acts_ff = self._task_feedforward(x, task_id)
        error = targets - acts_ff[-1]                      # teaching signal at output

        # Jacobian‑vector products w.r.t. hidden activations
        J_eff = [None]
        for l in range(1, len(self.layers) + 1):  # hidden layers only
            jac = torch.autograd.grad(
                outputs=acts_ff[-1],
                inputs=acts_ff[l],
                grad_outputs=error,
                retain_graph=True)[0]
            J_eff.append(jac)

        eq_acts = [acts_ff[0]]  # keep input
        pseudo_loss = torch.tensor(0., device=device)

        # hidden layers
        for l in range(1, len(self.layers) + 1):
            psi = J_eff[l] / (acts_ff[l] + 1e-8)
            gamma = self._compute_gamma(l - 1, B, device)
            modulation = torch.exp(psi + gamma)
            r_eq = modulation * acts_ff[l]
            eq_acts.append(r_eq)

            # teaching signal for weight l‑1
            ts = (r_eq - acts_ff[l]).detach()
            v_lm1 = self.layers[l - 1](acts_ff[l - 1])  # pre‑activation
            pseudo_loss += (ts * v_lm1).sum()

        # output layer (task head)
        y_eq = self.output_heads[task_id](eq_acts[-1])
        eq_acts.append(y_eq)
        ts_out = (y_eq - acts_ff[-1]).detach()
        v_out = self.output_heads[task_id](acts_ff[-2])
        pseudo_loss += (ts_out * v_out).sum()

        return eq_acts, pseudo_loss

    # ------------------------------------------------------------------
    # dynamic inversion (iterative PI control)
    # ------------------------------------------------------------------
    def _task_dynamic_inversion(self, x: torch.Tensor, targets: torch.Tensor, task_id: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B, device = x.shape[0], x.device
        acts_ff = self._task_feedforward(x, task_id)
        r_states = [a.detach().clone() for a in acts_ff]  # include output

        u = torch.zeros_like(targets)
        u_int = torch.zeros_like(targets)
        self._last_modulation = []

        for _ in range(self.tmax):
            # PI controller
            error = r_states[-1] - targets
            u_int = u_int + self.dt * (error - self.alpha * u)
            u_next = u_int + self.k_p * error
            if torch.norm(u_next - u) < self.eps:
                break

            # J^T u for each hidden layer
            psis = [None]
            for l in range(1, len(self.layers) + 1):
                psi = torch.autograd.grad(r_states[-1], r_states[l], u_next,
                                           retain_graph=True, create_graph=True)[0]
                psis.append(psi)

            # update hidden states
            new_states = [r_states[0]]  # X stays
            for l in range(1, len(self.layers) + 1):
                v_ff = self.layers[l - 1](new_states[l - 1])
                r_ff = self.act(v_ff)
                psi_norm = psis[l] / (r_ff + 1e-8)
                gamma = self._compute_gamma(l - 1, B, device)
                mod = torch.exp(psi_norm + gamma)
                self._last_modulation.append(mod.detach())
                tau = self.taus[l - 1]
                r_new = r_states[l] + tau * (mod * r_ff - r_states[l])
                new_states.append(r_new)

            # output layer (head)
            y_new = self.output_heads[task_id](new_states[-1])
            new_states.append(y_new)

            r_states = [s.detach() for s in new_states]
            u = u_next.detach()

        # compute pseudo‑loss
        pseudo_loss = torch.tensor(0., device=device)
        for l in range(1, len(self.layers) + 1):
            ts = (r_states[l] - acts_ff[l]).detach()
            v_lm1 = self.layers[l - 1](acts_ff[l - 1])
            pseudo_loss += (ts * v_lm1).sum()
        # output
        ts_out = (r_states[-1] - acts_ff[-1]).detach()
        v_out = self.output_heads[task_id](acts_ff[-2])
        pseudo_loss += (ts_out * v_out).sum()

        return r_states, pseudo_loss

    # ------------------------------------------------------------------
    # public forward wrappers
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None,
                *, task_id: Optional[int] = None, use_efc: bool = True):
        if task_id is None:
            task_id = getattr(self, "current_task", 0)

        if self.training and targets is not None and use_efc:
            # convert to one‑hot if necessary
            if targets.dim() == 1:
                targets = F.one_hot(targets, self.output_heads[task_id].out_features).float()

            if self.use_dynamic_inversion:
                acts_eq, pseudo_loss = self._task_dynamic_inversion(x, targets, task_id)
            else:
                acts_eq, pseudo_loss = self._task_non_dynamic_inversion(x, targets, task_id)
            self._pseudo_loss = pseudo_loss
            return acts_eq[-1]
        else:
            # inference mode (no EFC)
            return self._task_feedforward(x, task_id)[-1]

    # ------------------------------------------------------------------
    # loss helpers ------------------------------------------------------
    # ------------------------------------------------------------------
    def compute_loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "_pseudo_loss"):
            return self._pseudo_loss
        return super().compute_loss(output, target)

    # ------------------------------------------------------------------
    # Fisher machinery (shared trunk only) ------------------------------
    # ------------------------------------------------------------------
    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p) for n, p in self.named_parameters() if p.requires_grad}
        self.eval()
        n_samples = 0
        for x, y in dataloader:
            x = x.to(next(self.parameters()).device)
            y = y.to(next(self.parameters()).device)
            logits = self.forward(x, task_id=self.current_task, use_efc=False)
            log_probs = F.log_softmax(logits, dim=1)
            if y.dim() == 1:
                y = F.one_hot(y, logits.size(1)).float()
            ll = (log_probs * y).sum()
            self.zero_grad()
            ll.backward()
            with torch.no_grad():
                for n, p in self.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.pow(2)
            n_samples += x.size(0)
        for n in fisher:
            fisher[n] /= n_samples
        return fisher

    def complete_task(self, dataloader):
        self.theta_star = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}
        if self.first_task:
            self.fisher = self._compute_fisher(dataloader)
            self.first_task = False
        else:
            new_fisher = self._compute_fisher(dataloader)
            for n in self.fisher:
                if n in new_fisher:
                    self.fisher[n] += new_fisher[n]