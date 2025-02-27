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
        self.tau = config.tau

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

        gammas = []
        for i, layer in enumerate(self.layers):
            gamma = self._compute_fisher_modulation(layer, i) if not self._first_task else 0.0
            gammas.append(gamma)

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
                layer.r_prev = r_current[i - 1] if i != 0 else self.input

                # Basal
                v_ff_current[i] = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)

                # Apical with teaching signal and Fisher modulation
                psi = torch.bmm(u_next.unsqueeze(1), Js[i]).squeeze()
                gamma = gammas[i]

                e_psi_gamma = torch.tanh(psi + gamma) + 1

                # Soma with modulation
                # tau = layer.tau # self.dt / self.time_constant_ratio
                v_current[i] += self.tau * (e_psi_gamma * v_ff_current[i] - v_current[i])

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
        self.psi_lr = config.psi_lr
        self.inner_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        
    def _dynamical_inversion(self):
        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        # Initialize activations
        v_ff_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]

        for i, layer in enumerate(self.layers):
            v_ff_current[i] = layer.linear_activations.detach().clone()
            v_current[i] = layer.linear_activations.detach().clone()
            r_current[i] = layer.activations.detach().clone()

        # Initialize psi
        psi_params = []
        for i, layer in enumerate(self.layers):
            psi = torch.zeros((self.bzs, layer.weights.shape[0]), requires_grad=True)
            psi_params.append(psi)

        # Initialize gamma
        gammas = []
        for i, layer in enumerate(self.layers):
            gamma = self._compute_fisher_modulation(layer, i) if not self._first_task else 0.0
            gammas.append(gamma)

        # Optimizer and convergence guard
        optimizer = torch.optim.SGD(psi_params, lr=self.psi_lr)

        converged_mask = torch.zeros(self.bzs, dtype=torch.bool)
        psi_prev = [psi.detach().clone() for psi in psi_params]

        for t in range(self.tmax - 1):
            # Stop if all batch elements have converged
            if converged_mask.all():
                break
            
            optimizer.zero_grad()

            for i in range(len(v_current)):
                v_current[i] = v_current[i].detach()
                r_current[i] = r_current[i].detach()
                v_ff_current[i] = v_ff_current[i].detach()

            # Forward pass with current psi values
            for i, layer in enumerate(self.layers):
                layer.r_prev = r_current[i-1] if i > 0 else self.input
                
                v_ff = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                
                # Compute e_psi_gamma
                psi = psi_params[i]
                gamma = gammas[i]
                e_psi_gamma = torch.tanh(psi + gamma) + 1
                
                # Compute activation with modulation
                r_current[i] = e_psi_gamma * layer.activation_fn(v_current[i])
                v_ff_current[i] = v_ff
            
            # Compute loss between current output and target
            loss = self.inner_loss_fn(r_current[-1], self.targets)
            
            # Backpropagate to compute gradients for psi parameters
            loss.backward(retain_graph=True)
            optimizer.step()

            psi_changes = [torch.norm(psi - psi_prev[i], dim=1) for i, psi in enumerate(psi_params)]
            max_psi_change = torch.stack(psi_changes).max()  # Max change per batch element
            converged_mask |= max_psi_change < self.eps
            psi_prev = [psi.detach().clone() for psi in psi_params]

        # Steady-state values per layer - save final values
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i].detach().clone()
            layer.r_ff = layer.activation_fn(v_ff_current[i]).detach().clone()
            layer.r_prev = rs[i]
            rs.append(layer.r)