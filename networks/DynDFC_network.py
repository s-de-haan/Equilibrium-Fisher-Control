import torch
import torch.nn as nn
from networks.network_interface import *
from networks.layers import DFC_layer
from networks.activation_function import *


class DynDFC_network(Network, JacobianInterface, WdynInterface):
    """
    Dynamic Deep Feedback Control (DynDFC) network.

    This model follows the same structural principles as EFC_network_v5,
    while extending it with:
    - A full PID controller (Proportional, Integral, Derivative terms)
    - Learned top-down feedback weights (W_dyn)
    - Additive modulation of activity using both W_dyn and Q · u feedback
    - A standard DFC update rule based on the difference between steady-state
      and feedforward activity
    """

    def __init__(self, config, name="DynDFC_network"):
        super().__init__(DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)

        # Top-down feedback weights per hidden layer (not for output layer)
        self.W_dyn = nn.ParameterList()
        for l in range(len(self.layers) - 1):
            shape = (self.layer_sizes[l], self.layer_sizes[l + 1])
            self.W_dyn.append(nn.Parameter(torch.zeros(*shape), requires_grad=False))

        # Learning rates and controller parameters
        self.eta_dyn = config.get("eta_dyn", 1e-3)
        self.eta_ff = config.get("eta_ff", self.lr)
        self.k_i = config.get("k_i", 0.0)
        self.k_d = config.get("k_d", 0.0)
        self.sample_wise = config.get("sample_wise", True)  # Enable sample-wise updates

    @torch.no_grad()
    def _dynamical_inversion(self):
        """
        Dynamically evolve the network toward steady state using PID controller feedback
        and learned dynamic feedback. Combines both feedback streams in an additive modulation.
        """
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        u_current = torch.zeros((self.bzs, self.output_size))
        error_integral = torch.zeros((self.bzs, self.output_size))
        prev_output = self.layers[-1].r.clone()

        # Store initial feedforward activations (used for learning after convergence)
        for layer in self.layers:
            layer.r_ff = layer.r.clone()

        for t in range(1, self.tmax):
            output = self.layers[-1].r
            error = output - self.targets
            error_integral += self.dt * error
            a_dot = (output - prev_output) / self.dt
            prev_output = output.clone()

            # PID control signal
            _, Js = self._calculate_full_jacobian()
            J_top = Js[-1]
            u_t = self.k_p * error + self.k_i * error_integral + self.k_d * torch.bmm(J_top, a_dot.unsqueeze(2)).squeeze(2)
            q_u = torch.matmul(self.Q, u_t.T).T

            for i, layer in enumerate(self.layers):
                r_prev = self.layers[i - 1].r if i != 0 else self.input
                layer.r_prev = r_prev
                layer.v_ff = r_prev @ layer.weights.T + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                if i < len(self.layers) - 1:
                    feedback_dyn = torch.matmul(self.W_dyn[i], self.layers[i + 1].r.T).T
                    controller_mod = torch.matmul(Js[i].transpose(1, 2), q_u.unsqueeze(2)).squeeze(2)
                    combined_signal = feedback_dyn + controller_mod
                    modulation = torch.tanh(combined_signal) + 1
                    self._update_W_dyn(i, feedback_dyn, q_u)
                else:
                    modulation = torch.ones_like(layer.r_ff)

                layer.r += self.dt / self.time_constant_ratio * (modulation * layer.r_ff - layer.r)

            converged_mask |= torch.norm(u_t - u_current, dim=1) < self.eps
            if converged_mask.all():
                break
            u_current = u_t

    @torch.no_grad()
    def _update_W_dyn(self, l, feedback_dyn, q_u):
        """
        Hebbian-style update for dynamic feedback weights during convergence.

        Parameters:
        - l: current layer index
        - feedback_dyn: top-down signal from W_dyn[l] * a[l+1]
        - q_u: full controller signal projected through Q
        """
        r_next = self.layers[l + 1].r.detach()
        modulation_abs = feedback_dyn.detach().abs()
        hebbian_update = torch.bmm(modulation_abs.unsqueeze(2), r_next.unsqueeze(1))
        delta = hebbian_update.mean(dim=0) * self.eta_dyn
        self.W_dyn[l].data += delta

    @torch.no_grad()
    def update_weights(self):
        """
        Standard DFC learning rule:
        Update weights using the difference between steady-state and feedforward activations.
        Can be performed either sample-wise or batch-wise.
        """
        for i, layer in enumerate(self.layers):
            delta = layer.r - layer.r_ff
            input_ff = layer.r_prev

            if self.sample_wise:
                for b in range(delta.shape[0]):
                    dw = torch.ger(delta[b], input_ff[b])
                    layer.weights.data += self.eta_ff * dw
                    layer.bias.data += self.eta_ff * delta[b]
            else:
                grad_w = torch.bmm(delta.unsqueeze(2), input_ff.unsqueeze(1)).mean(dim=0)
                grad_b = delta.mean(dim=0)
                layer.weights.data += self.eta_ff * grad_w
                layer.bias.data += self.eta_ff * grad_b

