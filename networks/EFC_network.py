import torch

from networks.network_interface import *
from networks.layers import DFC_layer
from networks.activation_function import *

class EFC_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network"):
        Network.__init__(self, DFC_layer, mReLU, mLinear, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc  # Fisher preservation coefficient

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

            error = self.targets - r_current[-1]
            
            # Proportional and integral (PI) control
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps

            _, Js = self._calculate_full_jacobian()

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                r_previous = r_current[i - 1] if i != 0 else self.input

                # Basal
                v_ff_current[i] = r_previous.mm(layer.weights.t()) + layer.bias.unsqueeze(0)

                # Apical with teaching signal and Fisher modulation
                psi = torch.bmm(u_next.unsqueeze(1), Js[i]).squeeze()
                gamma = self._compute_fisher_modulation(layer, i) if not self._first_task else 0.0
                e_psi_gamma = torch.exp(psi + gamma)

                if not self._first_task and t == 10 and i == 1:
                    print("psi: ",torch.norm(psi), "gamma: ", torch.norm(gamma))

                if i == len(self.layers) - 1:
                    e_psi_gamma = torch.where(v_ff_current[i] > 0, e_psi_gamma, 1 / e_psi_gamma)

                # Soma with modulation
                tau = self.dt / self.time_constant_ratio
                v_current[i] += tau * (e_psi_gamma * v_ff_current[i] - v_current[i])

                layer.activation_fn.set_m(e_psi_gamma)
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
        """Compute Fisher-based modulation for parameter preservation"""
        gamma = 0.0
        for n, p in layer.named_parameters():
            name = f'layers.{i}.{n}'
            if p.requires_grad:
                gamma += self.beta * torch.sum(self._fisher[name] * (p - self._means[name]))
                
        return gamma

    def complete_task(self, dataloader):
        """Store parameters and compute Fisher matrix at task completion"""
        if self._first_task:
            self._first_task = False
        
        # Store current parameter values
        self._means = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                self._means[n] = p.data.clone()
        
        # Compute Fisher Information Matrix
        self._fisher = self._calculate_fisher(dataloader)