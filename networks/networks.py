import torch
import torch.nn as nn

from networks.network_interface import JacobianInterface, NetworkInterface
from networks.layers import DFC_layer, BP_layer

# TODO: Put accuracy into Epoch: 001, Train loss ...
# TODO: Remove double logging of Epoch: 001, Train loss ...
# TODO: config.save (clean /outputs/)
# TODO: fix config logging

# TODO: Is the non-dynamical inversion the same for a multiplicative rule?
# TODO: Jacobian is now calculated with ReLU derivative, not mReLU


class DFC_Mult_network(JacobianInterface):
    def __init__(self, config, name="DFC_Mult_network") -> None:
        super().__init__(DFC_layer, nn.ReLU, config, name)

    @torch.no_grad()
    def _non_dynamical_inversion(self):
        J, _ = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self.targets - self.y_hat
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.matmul(J, J_T) + self.alpha * torch.eye(J.shape[1]), error
        )

        psi = torch.matmul(J_T, u).squeeze(-1)
        psis = torch.tensor_split(
            psi,
            torch.cumsum(torch.tensor(self.layer_sizes[:-1]), dim=0).cpu(),
            dim=1,
        )

        rs = [self.input]

        for i, layer in enumerate(self.layers):
            v_ff = torch.matmul(rs[i], layer.weights.t())
            v_ff += layer.bias.unsqueeze(0).expand_as(v_ff)
            v = v_ff

            r_ff = layer.activation_fn(v_ff)
            activation_result = layer.activation_fn(v)
            if i == len(self.layers) - 1:  # linear output layer
                r = torch.where(  # if negative do *e^-psi if positive do *e^psi
                    activation_result > 0,
                    activation_result * torch.exp(psis[i]),
                    activation_result * torch.exp(-psis[i]),
                )
            else:
                r = activation_result * torch.exp(psis[i])
            rs.append(r)

            layer.v_ff = v_ff
            layer.v = v
            layer.e_psi = torch.exp(psis[i])

            layer.r = r
            layer.r_ff = r_ff
            layer.r_prev = rs[i]


class DFC_network(JacobianInterface):
    def __init__(self, config, name="DFC_network") -> None:
        super().__init__(DFC_layer, nn.ReLU, config, name)

    def _non_dynamical_inversion(self):
        J, _ = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self.targets - self.y_hat
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.matmul(J, J_T) + self.alpha * torch.eye(J.shape[1]), error
        )

        delta_v = torch.matmul(J_T, u).squeeze(-1)
        delta_vs = torch.tensor_split(
            delta_v,
            torch.cumsum(torch.tensor(self.layer_sizes[:-1]), dim=0).cpu(),
            dim=1,
        )

        rs = [self.input]

        for i, layer in enumerate(self.layers):
            v_ff = torch.matmul(rs[i], layer.weights.t())
            v_ff += layer.bias.unsqueeze(0).expand_as(v_ff)
            v = v_ff + delta_vs[i]

            r_ff = layer.activation_fn(v_ff)
            r = layer.activation_fn(v)
            rs.append(r)

            layer.v_ff = v_ff
            layer.v = v
            layer.delta_v = delta_vs[i]

            layer.r = r
            layer.r_ff = r_ff
            layer.r_prev = rs[i]

    @torch.no_grad()
    def _dynamical_inversion(self):
        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        v_fb_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_ff_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        u_current = torch.zeros((self.bzs, self.output_size))
        u_int_current = torch.zeros((self.bzs, self.output_size))

        for i, layer in enumerate(self.layers):
            v_ff_current[i] = layer.linear_activations
            v_current[i] = layer.linear_activations
            r_current[i] = layer.activations

        # Controller loop # TODO: while u(t) not converged
        _, Jis = self._calculate_full_jacobian()
        for _ in range(self.tmax - 1):
            error = self.targets - r_current[-1]

            # Proportional and integral (PI) control.
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                r_previous = r_current[i - 1] if i != 0 else self.input

                # Forward
                a = r_previous.mm(layer.weights.t())
                a += layer.bias.unsqueeze(0).expand_as(a)
                v_ff_current[i] = a

                # Apical input (Ju)
                v_fb_current[i] = torch.matmul(u_next.unsqueeze(1), Jis[i]).squeeze()

                # Soma with apical
                v_current[i] += (self.dt / self.time_constant_ratio) * (
                    v_fb_current[i] + v_ff_current[i] - v_current[i]
                )
                r_current[i] = layer.activation_fn(v_current[i])
                layer.linear_activations = v_current[i]
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


class BP_network(NetworkInterface):
    def __init__(self, config, name="BP_network") -> None:
        super().__init__(BP_layer, nn.ReLU, config, name)
