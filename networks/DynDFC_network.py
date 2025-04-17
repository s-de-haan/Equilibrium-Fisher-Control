import torch
import torch.nn as nn
from networks.network_interface import *
from networks.layers import DFC_layer
from networks.activation_function import *


class DynDFC_network(Network, JacobianInterface, WdynInterface):
    """
    Dynamic Deep Feedback Control (DynDFC) network.

    Implements:
    - PID controller feedback
    - Feedback modulation weights W_dyn updated during convergence
    - Feedforward weights W updated during convergence (gated)
    - Convergence based on output activity change (not controller signal)
    - Inference via dynamics using W_dyn, stopping on output convergence
    """

    def __init__(self, config, name="DynDFC_network"):
        super().__init__(DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)

        self.W_dyn = nn.ParameterList()
        for l in range(len(self.layers) - 1):
            shape = (self.layer_sizes[l], self.layer_sizes[l + 1])
            self.W_dyn.append(nn.Parameter(torch.zeros(*shape), requires_grad=False))

        self.eta_dyn = config.get("eta_dyn", 1e-3)
        self.eta_ff = config.get("eta_ff", self.lr)
        self.k_i = config.get("k_i", 0.0)
        self.k_d = config.get("k_d", 0.0)

    @torch.no_grad()
    def _dynamical_inversion(self):
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        output_prev = self.layers[-1].r.clone()
        error_integral = torch.zeros((self.bzs, self.output_size))

        for layer in self.layers:
            layer.r_ff = layer.r.clone()
            layer.r_prev_state = layer.r.clone()

        for t in range(1, self.tmax):
            output = self.layers[-1].r
            error = output - self.targets
            error_integral += self.dt * error
            a_dot_output = (output - output_prev) / self.dt
            output_prev = output.clone()

            _, Js = self._calculate_full_jacobian()
            J_top = Js[-1]
            u_t = self.k_p * error + self.k_i * error_integral + self.k_d * torch.bmm(J_top, a_dot_output.unsqueeze(2)).squeeze(2)
            q_u = torch.matmul(self.Q, u_t.T).T

            for i, layer in enumerate(self.layers):
                r_prev = self.layers[i - 1].r if i != 0 else self.input
                layer.r_prev = r_prev
                layer.v_ff = r_prev @ layer.weights.T + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                a_dot_l = (layer.r - layer.r_prev_state) / self.dt
                layer.r_prev_state = layer.r.clone()

                if i < len(self.layers) - 1:
                    feedback_dyn = torch.matmul(self.W_dyn[i], self.layers[i + 1].r.T).T
                    controller_mod = torch.matmul(Js[i].transpose(1, 2), q_u.unsqueeze(2)).squeeze(2)
                    combined_signal = feedback_dyn + controller_mod
                    modulation = torch.tanh(combined_signal) + 1

                    self._update_W_dyn(i, a_dot_l, self.layers[i + 1].r)
                    self._update_forward_weights(i, a_dot_l, r_prev)
                else:
                    modulation = torch.ones_like(layer.r_ff)

                layer.r += self.dt / self.time_constant_ratio * (modulation * layer.r_ff - layer.r)

            converged_mask |= torch.norm(a_dot_output, dim=1) < self.eps
            if converged_mask.all():
                break

    @torch.no_grad()
    def _update_W_dyn(self, l, a_dot, r_next):
        for b in range(a_dot.shape[0]):
            if a_dot[b].norm() > self.eps:
                delta = torch.ger(a_dot[b], r_next[b])
                self.W_dyn[l].data += self.eta_dyn * delta

    @torch.no_grad()
    def _update_forward_weights(self, l, a_dot, input_ff):
        for b in range(a_dot.shape[0]):
            if self.eps < a_dot[b].norm() < 5 * self.eps:
                dw = torch.ger(a_dot[b], input_ff[b])
                self.layers[l].weights.data += self.eta_ff * dw
                self.layers[l].bias.data += self.eta_ff * a_dot[b]

    @torch.no_grad()
    def update_weights(self):
        pass  # Feedforward weights are now updated during convergence

    @torch.no_grad()
    def dynamic_inference(self, x):
        self.eval()
        self.input = x
        self.bzs = x.shape[0]

        for layer in self.layers:
            layer.r = layer.activation_fn(layer.r_prev @ layer.weights.T + layer.bias.unsqueeze(0))
            layer.r_prev_state = layer.r.clone()

        output_prev = self.layers[-1].r.clone()
        for t in range(self.tmax):
            output = self.layers[-1].r
            a_dot_output = (output - output_prev) / self.dt
            output_prev = output.clone()

            _, Js = self._calculate_full_jacobian()
            u_t = self.k_p * (output - self.targets)
            q_u = torch.matmul(self.Q, u_t.T).T

            for i, layer in enumerate(self.layers):
                r_prev = self.layers[i - 1].r if i != 0 else self.input
                layer.v_ff = r_prev @ layer.weights.T + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                if i < len(self.layers) - 1:
                    feedback_dyn = torch.matmul(self.W_dyn[i], self.layers[i + 1].r.T).T
                    controller_mod = torch.matmul(Js[i].transpose(1, 2), q_u.unsqueeze(2)).squeeze(2)
                    modulation = torch.tanh(feedback_dyn + controller_mod) + 1
                else:
                    modulation = torch.ones_like(layer.r_ff)

                layer.r += self.dt / self.time_constant_ratio * (modulation * layer.r_ff - layer.r)

            if torch.norm(a_dot_output, dim=1).max() < self.eps:
                break

        return self.layers[-1].r
