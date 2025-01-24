import torch
import torch.nn as nn
import random

from networks.network_interface import NetworkInterface
from networks.layers import DFC_layer, BP_layer


class SSAInterface(NetworkInterface):
    def __init__(self, layer_class, activation_fn, config, name) -> None:
        super().__init__(layer_class, activation_fn, config, name)

        self._target_lr = config.target_lr
        self._alpha_di = config.alpha_di

    def backward(self, y):
        self._set_targets(y)
        self._non_dynamical_inversion()

        for layer in self.layers:
            layer.backward()

    def _set_targets(self, y):
        """MSE loss solution"""
        self.targets = (1 - 2 * self._target_lr) * self.y_hat + 2 * self._target_lr * y
        if random.random() < 0.005:
            print("RANDOM:", self.y_hat.mean())

    def _calculate_full_jacobian(self):
        Js = []

        activations_derivatives = [
            layer.activation_derivative(layer.linear_activations)
            for layer in self.layers
        ]
        bsz = self.layers[0].activations.shape[0]

        output_sz = self.layers[-1].out_features

        # Last layer
        Js.append(
            activations_derivatives[-1].view(bsz, output_sz, 1)
            * torch.eye(output_sz).repeat(bsz, 1, 1)
        )
        # Rest of the layers
        for i in range(len(self.layers) - 2, -1, -1):
            J = activations_derivatives[i].unsqueeze(1) * torch.matmul(
                Js[-1], self.layers[i + 1].weights
            )
            Js.append(J)

        Js.reverse()

        return torch.cat(Js, dim=2)


class DFC_SSA_Mult_network(SSAInterface):
    def __init__(self, config, name="DFC_SSA_Mult_network") -> None:
        super().__init__(DFC_layer, nn.ReLU, config, name)

        self._target_lr = config.target_lr
        self._alpha_di = config.alpha_di

        # TODO: Is the non-dynamical inversion the same for a multiplicative rule?
        # TODO: Jacobian is now calculated with ReLU derivative, not mReLU

    def _non_dynamical_inversion(self):
        J = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self.targets - self.y_hat
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.matmul(J, J_T) + self._alpha_di * torch.eye(J.shape[1]), error
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


class DFC_SSA_network(SSAInterface):
    def __init__(self, config, name="DFC_SSA_network") -> None:
        super().__init__(DFC_layer, nn.ReLU, config, name)

    def _non_dynamical_inversion(self):
        J = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self.targets - self.y_hat
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.matmul(J, J_T) + self._alpha_di * torch.eye(J.shape[1]), error
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


class BP_network(NetworkInterface):
    def __init__(self, config, name="BP_network") -> None:
        super().__init__(BP_layer, nn.ReLU, config, name)
