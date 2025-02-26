import torch

from networks.network_interface import *
from networks.layers import DFC_layer
from networks.activation_function import *

class EFC_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.clamp = config.clamp

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

            _, Js = self._calculate_full_jacobian()

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                layer.r_previous = r_current[i - 1] if i != 0 else self.input

                # Basal
                v_ff_current[i] = layer.r_previous.mm(layer.weights.t()) + layer.bias.unsqueeze(0)

                # Apical with teaching signal and Fisher modulation
                psi = torch.bmm(u_next.unsqueeze(1), Js[i]).squeeze()
                gamma = self._compute_fisher_modulation(layer, i) if not self._first_task else 0.0
                if self.clamp:
                    if not self._first_task: # Maximal effect of gamma is to undo psi, i.e. back to baseline
                        scaling_factor = torch.abs(psi).mean()
                        gamma = torch.tanh(gamma / scaling_factor) * scaling_factor
                        torch.clamp(gamma, min=-torch.abs(psi), max=torch.abs(psi))

                e_psi_gamma = torch.exp(psi + gamma)

                # Soma with modulation
                tau = layer.tau # self.dt / self.time_constant_ratio
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


class EFC_BP_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_BP_network"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.tau = config.tau
        
    def _dynamical_inversion(self):
        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        # Initialize activations
        v_ff_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]

        for i, layer in enumerate(self.layers):
            v_ff_current[i] = layer.linear_activations
            v_current[i] = layer.linear_activations
            r_current[i] = layer.activations
            layer.activation_fn.reset_m()

        # Initialize psi parameters (with gradients enabled)
        psi_params = []
        for i, layer in enumerate(self.layers):
            psi = torch.zeros((self.bzs, layer.weights.shape[0]), requires_grad=True)
            psi_params.append(psi)
        
        optimizer = torch.optim.Adam(psi_params, lr=self.dt)

        converged_mask = torch.zeros(self.bzs, dtype=torch.bool)
        prev_loss = torch.full((self.bzs,), float('inf'))

        for t in range(self.tmax - 1):
            # Stop if all batch elements have converged
            if converged_mask.all():
                break
            
            optimizer.zero_grad()
            
            # Forward pass with current psi values
            for i, layer in enumerate(self.layers):
                r_prev = r_current[i-1] if i > 0 else self.input
                
                v_ff = r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                
                # Compute e_psi_gamma
                psi = psi_params[i]
                gamma = self._compute_fisher_modulation(layer, i) if not self._first_task else 0.0
                e_psi_gamma = torch.exp(psi + gamma)
                
                # Integrate soma potential with modulation
                v_current[i] += self.tau * (e_psi_gamma * v_ff - v_current[i])
                
                # Compute activation with modulation
                r_current[i] = layer.activation_fn(v_current[i])
                v_ff_current[i] = v_ff
                
                layer.activation_fn.set_m(e_psi_gamma)
                layer.r_previous = r_prev
            
            # Compute error between current output and target
            error = self._compute_error(r_current[-1], self.targets)
            loss = torch.sum(torch.norm(error, dim=1))
            
            current_loss = torch.norm(error, dim=1)
            converged_mask |= torch.abs(current_loss - prev_loss) < self.eps
            prev_loss = current_loss.detach()
            
            # Backpropagate to compute gradients for psi parameters
            if not converged_mask.all():
                loss.backward()
                optimizer.step()

        # Steady-state values per layer - save final values
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])