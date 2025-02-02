import torch
import torch.nn as nn

from networks.network_interface import JacobianInterface
from networks.layers import DFC_layer
from networks.activation_function import *


class DFC_Mult_network(JacobianInterface):
    def __init__(self, config, name="DFC_Mult_network") -> None:
        super().__init__(DFC_layer, mReLU, mLinear, config, name)

    @torch.no_grad()
    def _non_dynamical_inversion(self):
        J, _ = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self.targets - self.y_hat
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.bmm(J, J_T) + self.alpha * torch.eye(J.shape[1]), error
        )

        psi = torch.bmm(J_T, u).squeeze(-1)
        psis = torch.split(psi, self.layer_sizes, dim=1)

        rs = [self.input]

        for i, layer in enumerate(self.layers):
            v_ff = torch.mm(rs[i], layer.weights.t())
            v_ff += layer.bias.unsqueeze(0).expand_as(v_ff)
            v = v_ff
            r_ff = layer.activation_fn(v_ff)

            e_psi = torch.exp(torch.abs(psis[i])) 
            if i == len(self.layers) - 1:
                e_psi = torch.where(r_ff > 0, e_psi, 1 / e_psi)

            layer.activation_fn.set_m(e_psi)
            r = layer.activation_fn(v)
            rs.append(r)

            layer.v_ff = v_ff
            layer.v = v
            layer.e_psi = e_psi

            layer.r = r
            layer.r_ff = r_ff
            layer.r_prev = rs[i]

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

        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)

        for t in range(self.tmax - 1):
            # Stop if converged
            if converged_mask.all():
                break

            error = self.targets - r_current[-1]
            
            # Proportional and integral (PI) control.
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps

            _, Js = self._calculate_full_jacobian()

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                r_previous = r_current[i - 1] if i != 0 else self.input

                # Basal and apical
                v_ff_current[i] = r_previous.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                e_psi = torch.exp(torch.bmm(u_next.unsqueeze(1), Js[i]).squeeze())
                if i == len(self.layers) - 1: # Correct for linear output layer
                    e_psi = torch.where(v_ff_current[i] > 0, e_psi, 1 / e_psi)

                # Soma with apical
                tau = self.dt / self.time_constant_ratio
                v_current[i] += tau * (e_psi * v_ff_current[i] - v_current[i])

                layer.activation_fn.set_m(e_psi)
                r_current[i] = layer.activation_fn(v_current[i])

                layer.linear_activations = v_current[i]
                layer.activations = r_current[i]
                print(layer.activations)

            u_int_current = u_int_next
            u_current = u_next

        # Steady-state values per layer
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])






