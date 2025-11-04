import torch
from networks.network_interface import *
from networks.layers import BP_layer
from networks.activation_function import ReLU, Linear
from tqdm import tqdm
import torch.autograd.functional as F


class EHC_network(Network, FisherInterface):
    def __init__(self, config, name="EHC_network"):
        Network.__init__(self, BP_layer, ReLU, Linear, config, name)
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
        current_hessian = self._calculate_hessian(dataloader)
        self._theta_star = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}

        if self._first_task:
            self._hessian = current_hessian
            self._first_task = False
        else:
            for n in self._hessian:
                self._hessian[n] += current_hessian[n]

    def _calculate_hessian(self, loader):
        params = [p for p in self.parameters() if p.requires_grad]
        p = sum(param.numel() for param in params)
        H = torch.zeros(p, p, dtype=torch.float32, device=self.device)
        total_samples = 0

        pbar = tqdm(total=len(loader), desc="Hessian", leave=False)

        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            b = x.size(0)

            def loss_func(params_flat):
                offset = 0
                for param in params:
                    numel = param.numel()
                    param.data.copy_(params_flat[offset:offset + numel].view(param.shape))
                    offset += numel
                out = self(x)
                return self.loss_fn(out, y.argmax(dim=1)) / b

            params_flat = torch.cat([param.data.flatten() for param in params])
            H_batch = F.hessian(loss_func, (params_flat,))[0][0]
            H += H_batch * b
            total_samples += b
            pbar.update(1)

        pbar.close()
        if total_samples > 0:
            H /= total_samples
        return H