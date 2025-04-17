import torch
import torch.nn as nn
from networks.network_interface import Network, JacobianInterface, WdynInterface
from networks.layers import DFC_layer
from networks.activation_function import *


class DynDFC_network(Network, JacobianInterface, WdynInterface):
    """
    Dynamic Deep Feedback Control (DynDFC) network.

    Implements:
    - PID controller feedback
    - Feedback modulation weights W_dyn updated via gated Hebbian
    - Feedforward weights W updated via gated Hebbian
    - Convergence-based plasticity using activity change
    """

    def __init__(self, config, name="DynDFC_network"):
        super().__init__(DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)

        # Initialize dynamic feedback weights
        self.W_dyn = nn.ParameterList()
        for l in range(len(self.layers) - 1):
            shape = (self.layer_sizes[l], self.layer_sizes[l + 1])
            self.W_dyn.append(nn.Parameter(torch.zeros(*shape), requires_grad=False))

        # Learning rates
        self.eta_dyn = config.get("eta_dyn", 1e-3)
        self.eta_ff = config.get("eta_ff", self.lr)

        # Controller gains
        self.k_p = config.get("k_p", 1.0)
        self.k_i = config.get("k_i", 0.0)
        self.k_d = config.get("k_d", 0.0)

    def compute_gates(self, a_old, a_new, eps, max_factor=5.0):
        diff = (a_new - a_old).abs().mean(dim=1)
        g_dyn = (diff > eps).float()
        g_ff = ((diff > eps) & (diff < max_factor * eps)).float()
        return g_dyn, g_ff

    @torch.no_grad()
    def _get_jacobian_layer(self, layer_idx):
        """
        Returns the transpose of the Jacobian for layer 'layer_idx', shape [batch, in, out].
        """
        _, Js = self._calculate_full_jacobian()
        return Js[layer_idx].transpose(1, 2)

    @torch.no_grad()
    def _dynamical_inversion(self):
        bsz = self.bzs
        output_prev = self.layers[-1].r.clone()
        for layer in self.layers:
            layer.r_prev_state = layer.r.clone()

        for t in range(self.tmax):
            output = self.layers[-1].r
            error = output - self.targets
            a_dot_out = (output - output_prev) / self.dt
            output_prev = output.clone()

            # PID controller (P + D)
            u = self.k_p * error + self.k_d * a_dot_out
            q_u = torch.matmul(self.Q, u.T).T

            for i, layer in enumerate(self.layers):
                r_in = self.input if i == 0 else self.layers[i - 1].r
                layer.v_ff = r_in @ layer.weights.T + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                a_dot = (layer.r - layer.r_prev_state) / self.dt
                layer.r_prev_state = layer.r.clone()

                if i < len(self.layers) - 1:
                    # Dynamic feedback and controller modulation
                    fb = (self.W_dyn[i] @ self.layers[i + 1].r.T).T
                    ctrl = torch.matmul(self._get_jacobian_layer(i), q_u.unsqueeze(2)).squeeze(2)
                    modulation = torch.tanh(fb + ctrl) + 1.0

                    # Gated Hebbian updates
                    g_dyn, g_ff = self.compute_gates(layer.r_prev_state, layer.r, self.eps)
                    self._update_W_dyn(i, a_dot, self.layers[i + 1].r, g_dyn)
                    self._update_forward_weights(i, a_dot, r_in, g_ff)
                else:
                    modulation = torch.ones_like(layer.r_ff)

                # Euler integration of dynamics
                layer.r += self.dt / self.time_constant_ratio * (modulation * layer.r_ff - layer.r)

            # Convergence check on output
            if torch.norm(a_dot_out, dim=1).max() < self.eps:
                break

    @torch.no_grad()
    def _update_W_dyn(self, layer_idx, a_dot, r_next, gate):
        for b in range(gate.shape[0]):
            if gate[b] > 0:
                delta = torch.ger(a_dot[b], r_next[b])
                self.W_dyn[layer_idx].data += self.eta_dyn * delta

    @torch.no_grad()
    def _update_forward_weights(self, layer_idx, a_dot, r_in, gate):
        layer = self.layers[layer_idx]
        for b in range(gate.shape[0]):
            if gate[b] > 0:
                dw = torch.ger(a_dot[b], r_in[b])
                layer.weights.data += self.eta_ff * dw
                layer.bias.data += self.eta_ff * a_dot[b]

    @torch.no_grad()
    def update_weights(self):
        # Not used: updates occur during dynamics
        pass

    @torch.no_grad()
    def dynamic_inference(self, x):
        self.eval()
        self.input = x
        self.bzs = x.shape[0]
        for layer in self.layers:
            r_in = self.input if layer is self.layers[0] else prev_r
            layer.r = layer.activation_fn(r_in @ layer.weights.T + layer.bias.unsqueeze(0))
            prev_r = layer.r.clone()

        output_prev = self.layers[-1].r.clone()
        for t in range(self.tmax):
            output = self.layers[-1].r
            a_dot_out = (output - output_prev) / self.dt
            output_prev = output.clone()

            u = self.k_p * (output - self.targets)
            q_u = torch.matmul(self.Q, u.T).T
            for i, layer in enumerate(self.layers):
                r_in = self.input if i == 0 else self.layers[i - 1].r
                layer.r_ff = layer.activation_fn(r_in @ layer.weights.T + layer.bias.unsqueeze(0))
                if i < len(self.layers) - 1:
                    fb = (self.W_dyn[i] @ self.layers[i + 1].r.T).T
                    ctrl = torch.matmul(self._get_jacobian_layer(i), q_u.unsqueeze(2)).squeeze(2)
                    modulation = torch.tanh(fb + ctrl) + 1.0
                else:
                    modulation = torch.ones_like(layer.r_ff)
                layer.r += self.dt / self.time_constant_ratio * (modulation * layer.r_ff - layer.r)

            if torch.norm(a_dot_out, dim=1).max() < self.eps:
                break

        return self.layers[-1].r
