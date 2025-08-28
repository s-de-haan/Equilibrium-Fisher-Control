import torch

from networks.network_interface import *
from networks.layers import *
from networks.activation_function import *

class EFC_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc

    @torch.no_grad()
    def _non_dynamical_inversion(self):
        # Calculate Jacobians for each layer
        Js = self._calculate_layerwise_jacobians()

        # Compute J_eff and gamma_eff with Q = J^T
        J_eff, gamma_eff, J_i, gamma_i = self._calculate_jeff_and_gammaeff()

        # Compute the output error
        delta_L_minus = self._compute_error(self.y_hat, self.targets)

        # Solve for u_star: (alpha I + J_eff) u_star = delta_L_minus - gamma_eff
        u_star = torch.linalg.solve(J_eff + self.alpha_I * torch.eye(J_eff.shape[1]), delta_L_minus - gamma_eff)

        # Compute the control signal for each layer
        Qu_i = [torch.bmm(J_i[i].transpose(1, 2), u_star.unsqueeze(-1)).squeeze(-1) for i in range(len(Js))]

        # Compute delta_r = (I - J_{i,i-1})^-1 (Qu*_i + gamma_i) for each layer and update r^* = r^- + delta_r
        delta_r_prev = torch.zeros_like(self.input)
        for i, layer in enumerate(self.layers):
            # (Qu*_i + γ_i) ⊙ r^-_i + J_{i,i-1} * Δr_{i-1} where J_{i,i-1} = φ'(W_i * r^-_{i-1}) ⊙ W_i
            # This is equivalent to φ'(pre_activation) ⊙ (W_i @ Δr_{i-1})
            delta_r_i = (Qu_i[i] + gamma_i[i]) * layer.r_ff + torch.matmul(Js[i], delta_r_prev.unsqueeze(-1)).squeeze(-1)
            delta_r_prev = delta_r_i

            layer.r = layer.r_ff + delta_r_i
            

    @torch.no_grad()
    def _dynamical_inversion(self):
        u_current = torch.zeros((self.bzs, self.output_size))
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)

        for t in range(self.tmax):
            error = self._compute_error(self.layers[-1].r, self.targets)
            
            # Proportional control
            u_next = self.k_p * error
            psis = self._calculate_psis(u_next)

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i-1].r if i != 0 else self.input
                layer.v_ff = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                # Apical with teaching signal and Fisher modulation
                psi = psis[i]
                gamma = self._compute_gamma(layer, i)
                if not self._first_task: # Maximal effect of gamma is to undo psi, i.e. back to baseline
                    scaling_factor = torch.abs(psi).mean()
                    gamma = torch.tanh(gamma / scaling_factor) * scaling_factor
                e_psi_gamma = torch.tanh(psi + gamma) + 1

                # Soma with modulation
                layer.r = layer.r + self.dt / self.time_constant_ratio * (e_psi_gamma * layer.r_ff - layer.r)

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all() and t > 10:
                break
            u_current = u_next