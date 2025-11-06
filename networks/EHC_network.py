import torch
from networks.network_interface import *
from networks.layers import BP_layer
from networks.activation_function import ReLU, Linear, Softplus
from tqdm import tqdm
import torch.autograd.functional as F


class EHC_network(Network, FisherInterface):
    def __init__(self, config, name="EHC_network"):
        Network.__init__(self, BP_layer, Softplus, Linear, config, name)
        FisherInterface.__init__(self)
        self.importance = config.importance_ewc
        self._theta_star = None
        self._hessian = None  # Full p x p Hessian

    def ehc_loss(self):
        if self._first_task or self._hessian is None:
            return torch.tensor(0.0, device=self.device)

        v = torch.cat(
            [(p - self._theta_star[n]).flatten()
            for n, p in self.named_parameters() if p.requires_grad]
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
        self._theta_star = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}

        if self._first_task:
            self._hessian = current_hessian
            self._first_task = False
        else:
            for n in self._hessian:
                self._hessian[n] += current_hessian[n]

    def _calculate_full_fisher(self, loader):
        params = [p for p in self.parameters() if p.requires_grad]
        p = sum(par.numel() for par in params)
        F = torch.zeros(p, p, dtype=torch.float32, device=self.device)
        total = 0

        self.eval()
        pbar = tqdm(total=len(loader), desc="Fisher", leave=False)

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            b = x.size(0)

            self.zero_grad()
            out = self(x)
            loss = self.loss_fn(out, y)
            loss.backward()

            grads = torch.cat([p.grad.flatten() for p in params])
            F += torch.outer(grads, grads) * (b ** 2)
            total += b
            pbar.update(1)
            
        pbar.close()
        F /= total
        return F