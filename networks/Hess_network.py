import torch
from networks.network_interface import *
from networks.layers import BP_layer
from networks.activation_function import ReLU, Linear, Softplus
from tqdm import tqdm
import torch.nn.functional as F
from torch.func import functional_call, vmap, grad


class Hess_network(Network, FisherInterface):
    def __init__(self, config, name="Hess_network"):
        Network.__init__(self, BP_layer, Softplus, Linear, config, name)
        FisherInterface.__init__(self)
        self.importance = config.importance_ewc
        self._theta_star = None
        self._hessian = None  # Full p x p Hessian

    def ehc_loss(self):
        if self._first_task or self._hessian is None:
            return torch.tensor(0.0, device=self.device)

        v = torch.cat(
            [
                (p - self._theta_star[n]).flatten()
                for n, p in self.named_parameters()
                if p.requires_grad
            ]
        )
        quad = 0.5 * (v @ self._hessian @ v)
        return self.importance * quad

    def backward(self, y):
        loss = self.loss_fn(self.y_hat, y)
        if not self._first_task:
            loss += self.ehc_loss()
        loss.backward()

    def complete_task(self, dataloader):
        # Use your _calculate_hessian from FisherInterface
        current_hessian = self._calculate_full_fisher(dataloader)
        self._theta_star = {
            n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad
        }

        if self._first_task:
            self._hessian = current_hessian
            self._first_task = False
        else:
            for n in self._hessian:
                self._hessian[n] += current_hessian[n]

    def _calculate_full_fisher(self, dataloader):
        """Compute full Fisher Information Matrix"""
        # Get flattened parameters
        params_dict = {n: p for n, p in self.named_parameters() if p.requires_grad}
        buffers = {n: b for n, b in self.named_buffers()}

        # Count total parameters
        param_count = sum(p.numel() for p in params_dict.values())
        fisher = torch.zeros(
            param_count, param_count, dtype=torch.float32, device=self.device
        )

        def compute_loss_single(params, buffers, x, y):
            """Compute log likelihood for a single sample"""
            output = functional_call(self, (params, buffers), (x.unsqueeze(0),))
            log_probs = F.log_softmax(output, dim=1)
            log_likelihood = (log_probs * y.unsqueeze(0)).sum()
            return log_likelihood

        # Create gradient function and vectorize it
        grad_fn = grad(compute_loss_single)
        grad_fn_vmap = vmap(grad_fn, in_dims=(None, None, 0, 0))

        self.eval()
        total_samples = 0
        pbar = tqdm(total=len(dataloader), desc="Hessian", leave=True)

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)

            # Compute per-sample gradients (parallelized across batch)
            per_sample_grads = grad_fn_vmap(params_dict, buffers, inputs, targets)

            # Flatten gradients for each sample
            # per_sample_grads is a dict with shape [batch_size, ...] for each param
            grads_flat = torch.stack(
                [
                    torch.cat(
                        [per_sample_grads[n][i].flatten() for n in params_dict.keys()]
                    )
                    for i in range(batch_size)
                ]
            )  # Shape: [batch_size, param_count]

            # Accumulate outer products
            fisher += torch.einsum("bi,bj->ij", grads_flat, grads_flat)

            total_samples += batch_size
            pbar.update(1)

        pbar.close()

        # Normalize
        fisher /= total_samples

        return fisher
