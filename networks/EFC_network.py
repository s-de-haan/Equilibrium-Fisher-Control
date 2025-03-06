import torch

from networks.network_interface import *
from networks.layers import DFC_layer
from networks.activation_function import *


"""
    "layers": [784, 400, 400, 2],
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 5,
        "mode": "di",  # or "di"
        "num_workers": 8,
        "loss_fn": "ce", # "mse"
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cuda",
        "output_dir": "./outputs",
        "seed": 0,
        "target_lr": 1.0, # needs to be < time_constant_ratio
        "alpha_di": 1e-4,
        "taus": [0.01, 0.008, 0.006],
        "time_constant_ratio": 0.2, # this param can be merged with dt_di
        "tmax_di": 500,
        "dt_di": 0.001,
        "k_p": 2.0,
        "eps": 1e-3, # there is an interplay between dt_di and eps
        "save": False,
        "importance_ewc": 1.0, # ewc params
        "beta_efc": 1000.0, # efc params
"""
class EFC_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.tau = config.taus

    @torch.no_grad()
    def _dynamical_inversion(self):
        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        v_ff_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        u_current = torch.zeros((self.bzs, self.output_size))
        u_int_current = torch.zeros((self.bzs, self.output_size))

        for i, layer in enumerate(self.layers):
            v_ff_current[i] = layer.linear_activations
            v_current[i] = layer.linear_activations
            r_current[i] = layer.activations
            layer.activation_fn.reset_m()

        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)

        for t in range(self.tmax - 1):
            # Stop if converged
            if converged_mask.all():
                break

            error = self._compute_error(r_current[-1], self.targets)
            
            # Proportional and integral (PI) control
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps

            psis = _calculate_psis(u_next)

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                layer.r_previous = r_current[i - 1] if i != 0 else self.input

                # Basal
                v_ff_current[i] = layer.r_previous.mm(layer.weights.t()) + layer.bias.unsqueeze(0)

                # Apical with teaching signal and Fisher modulation
                psi = psis[i] #= torch.bmm(u_next.unsqueeze(1), Js[i]).squeeze()
                gamma = self._compute_fisher_modulation(layer, i) 
                if not self._first_task: # Maximal effect of gamma is to undo psi, i.e. back to baseline
                    # scaling_factor = torch.abs(psi).mean()
                    # gamma = torch.tanh(gamma / scaling_factor) * scaling_factor
                    torch.clamp(gamma, min=-torch.abs(psi), max=torch.abs(psi))

                e_psi_gamma = torch.tanh(psi + gamma) + 1

                # Soma with modulation
                tau = self.tau[i] # self.dt / self.time_constant_ratio
                v_current[i] += tau * (e_psi_gamma * v_ff_current[i] - v_current[i])

                # layer.activation_fn.set_m(e_psi_gamma)
                r_current[i] = layer.activation_fn(v_current[i])

                layer.linear_activations = v_ff_current[i]
                layer.activations = r_current[i]

            u_int_current = u_int_next
            u_current = u_next

        # Steady-state values per layer
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])
    

    def _compute_fisher_modulation(self, layer, i):
        if self._first_task:
            return 0.0
        """Compute Fisher-based modulation for parameter preservation"""
        gamma = torch.zeros((self.bzs, layer.weights.shape[0]))
        fisher_norm = 0.0

        for n, p in layer.named_parameters():
            full_name = f'layers.{i}.{n}'
            if p.requires_grad:
                base_gamma = self._fisher[full_name] * (p - self._means[full_name])
                if 'weights' in n:
                    gamma += (layer.r_previous @ base_gamma.T)
                    fisher_norm += torch.sum(self._fisher[full_name]**2, dim=1)
                elif 'bias' in n:
                    gamma += base_gamma
                    fisher_norm += self._fisher[full_name]**2
        
        gamma = - self.beta * gamma / (torch.sqrt(fisher_norm) + 1e-8)
        
        return gamma