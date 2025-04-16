import torch

from networks.network_interface import *
from networks.layers import DFC_layer
from networks.activation_function import *

check_nan = lambda x: torch.isnan(x).any().item()

class DFC_network(Network, JacobianInterface):
    def __init__(self, config, name="DFC_network") -> None:
        Network.__init__(self, DFC_layer, ReLU, Linear, config, name)
        JacobianInterface.__init__(self, config)

    @torch.no_grad()
    def _non_dynamical_inversion(self):
        J, _ = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self._compute_error(self.y_hat, self.targets)
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.bmm(J, J_T) + self.alpha * torch.eye(J.shape[1]), error
        )

        delta_v = torch.bmm(J_T, u).squeeze(-1)
        delta_vs = torch.split(delta_v, self.layer_sizes, dim=1)

        rs = [self.input]

        for i, layer in enumerate(self.layers):
            v_ff = torch.bmm(rs[i], layer.weights.t())
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
        # Setup
        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        v_fb_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_ff_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod)) for lod in layer_out_dims]
        u_current = torch.zeros((self.bzs, self.output_size))
        u_int_current = torch.zeros((self.bzs, self.output_size))

        for i, layer in enumerate(self.layers):
            v_ff_current[i] = layer.v_ff
            v_current[i] = layer.v_ff
            r_current[i] = layer.r

        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)

        # Simulate tmax timesteps
        for _ in range(self.tmax - 1):
            # Stop if converged
            if converged_mask.all():
                break

            error = self._compute_error(r_current[-1], self.targets)
            
            # Proportional and integral (PI) control.
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps

            _, Js = self._calculate_full_jacobian()
            
            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                r_prev = r_current[i - 1] if i != 0 else self.input

                # Basal and apical
                v_ff_current[i] = r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                v_fb_current[i] = torch.bmm(Js[i].transpose(1, 2), u_next.unsqueeze(2)).squeeze(2)
                
                # Soma with apical
                tau = self.dt / self.time_constant_ratio
                v_current[i] += tau * (v_fb_current[i] + v_ff_current[i] - v_current[i])
                r_current[i] = layer.activation_fn(v_current[i])

                layer.v_ff = v_current[i]
                layer.r = r_current[i]

            u_int_current = u_int_next
            u_current = u_next

        # Steady-state values per layer
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])


class DFC_Mult_network(Network, JacobianInterface):
    def __init__(self, config, name="DFC_Mult_network") -> None:
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)

    @torch.no_grad()
    def _non_dynamical_inversion(self):
        J, _ = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self._compute_error(self.y_hat, self.targets)
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

            e_psi = torch.tanh(psis[i]) + 1

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
            v_ff_current[i] = layer.v_ff
            v_current[i] = layer.v_ff
            r_current[i] = layer.r
            layer.activation_fn.reset_modulation()

        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)

        for t in range(self.tmax - 1):
            # Stop if converged
            if converged_mask.all():
                break

            error = self._compute_error(r_current[-1], self.targets)
            
            # Proportional and integral (PI) control.
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps

            _, Js = self._calculate_full_jacobian()

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                r_prev = r_current[i - 1] if i != 0 else self.input

                # Basal
                v_ff_current[i] = r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                
                # Apical
                e_psi = torch.exp(torch.bmm(u_next.unsqueeze(1), Js[i]).squeeze())
                # if i == len(self.layers) - 1: # Correct for linear output layer
                #     e_psi = torch.where(v_ff_current[i] > 0, e_psi, 1 / e_psi)

                # Soma with apical
                tau = self.dt / self.time_constant_ratio
                v_current[i] += tau * (e_psi * v_ff_current[i] - v_current[i])

                layer.activation_fn.set_modulation(e_psi)
                r_current[i] = layer.activation_fn(v_current[i])

                layer.v_ff = v_ff_current[i]
                layer.r = r_current[i]

            u_int_current = u_int_next
            u_current = u_next

        # Steady-state values per layer
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])

"""
    "layers": [784, 400, 400, 2],
    "lr": 1e-3,
    "target_lr": 1.0,
    "dt_di": 0.0016,
    "time_constant_ratio": 0.2,
    "tmax_di": 500,
    "k_p": 1.0,
    "eps": 1e-4,
"""
class DFC_Mult_network_clean(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network_v4"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)

    @torch.no_grad()
    def _dynamical_inversion(self):
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        u_current = torch.zeros((self.bzs, self.output_size))

        for t in range(1, self.tmax):
            error = self._compute_error(self.layers[-1].r, self.targets)
            
            # Proportional control
            u_next = self.k_p * error
            psis = self._calculate_psis(u_next)

            # Forward pass
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i-1].r if i != 0 else self.input
                layer.v_ff = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)
                
                psi = psis[i]
                e_psi = torch.tanh(psi) + 1

                layer.r = layer.r + self.dt / self.time_constant_ratio * (e_psi * layer.r_ff - layer.r)

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all():
                break
            u_current = u_next

class DFC_Mult_network_dynamic_feedback(Network, JacobianInterface, WdynInterface):
    def __init__(self, config, name="DFC_Mult_network_dynamic_feedback"):
        super().__init__(DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)

        # Dynamic feedback weights: W_dyn[l] projects from layer l+1 to l
        self.W_dyn = nn.ParameterList()
        for l in range(len(self.layers) - 1):  # Skip output layer
            shape = (self.layer_sizes[l], self.layer_sizes[l + 1])
            W_dyn_l = nn.Parameter(torch.zeros(*shape), requires_grad=False)
            self.W_dyn.append(W_dyn_l)

        self.eta_dyn = config.get("eta_dyn", 1e-3)         # Learning rate for W_dyn
        self.eta_ff = config.get("eta_ff", self.lr)        # Learning rate for feedforward weights
        self.k_i = config.get("k_i", 0.0)                  # Integral gain
        self.k_d = config.get("k_d", 0.0)                  # Derivative gain

    @torch.no_grad()
    def _dynamical_inversion(self):
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        u_current = torch.zeros((self.bzs, self.output_size))
        error_integral = torch.zeros((self.bzs, self.output_size))
        prev_output = self.layers[-1].r.clone()

        for t in range(1, self.tmax):
            output = self.layers[-1].r
            error = output - self.targets
            error_integral += self.dt * error
            a_dot = (output - prev_output) / self.dt
            prev_output = output.clone()

            # PID controller
            _, Js = self._calculate_full_jacobian()
            J_top = Js[-1]
            u_t = self.k_p * error + self.k_i * error_integral + self.k_d * torch.bmm(J_top, a_dot.unsqueeze(2)).squeeze(2)
            q_u = torch.matmul(self.Q, u_t.T).T  # Apply Q once to full control signal

            # Forward pass with W_dyn modulation
            for i, layer in enumerate(self.layers):
                r_prev = self.layers[i - 1].r if i != 0 else self.input
                layer.r_prev = r_prev
                layer.v_ff = r_prev @ layer.weights.T + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                if i < len(self.layers) - 1:
                    feedback_dyn = torch.matmul(self.W_dyn[i], self.layers[i + 1].r.T).T
                    modulation = torch.tanh(feedback_dyn) + 1.0
                    self._update_W_dyn(i, modulation, q_u)  # Update W_dyn during convergence
                else:
                    modulation = torch.ones_like(layer.r_ff)

                layer.r = layer.r + self.dt / self.time_constant_ratio * (modulation * layer.r_ff - layer.r)

            # Convergence check
            converged_mask |= torch.norm(u_t - u_current, dim=1) < self.eps
            if converged_mask.all():
                break
            u_current = u_t

        # Final state update
        for i, layer in enumerate(self.layers):
            layer.r_prev = self.layers[i - 1].r if i != 0 else self.input
            layer.v_ff = layer.r_prev @ layer.weights.T + layer.bias.unsqueeze(0)
            layer.r_ff = layer.activation_fn(layer.v_ff)

        # Feedforward weight update (after convergence)
        self._update_weights()

    @torch.no_grad()
    def _update_W_dyn(self, l, modulation, q_u):
        r_next = self.layers[l + 1].r.detach()
        modulation_abs = modulation.detach().abs()
        q_u_detached = q_u.detach()

        hebbian_update = torch.bmm(modulation_abs.unsqueeze(2), r_next.unsqueeze(1))
        delta = hebbian_update.mean(dim=0) * self.eta_dyn
        self.W_dyn[l].data += delta

    @torch.no_grad()
    def _update_weights(self):
        J, _ = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self._compute_error(self.y_hat, self.targets).unsqueeze(2)
        u = torch.linalg.solve(
            torch.bmm(J, J_T) + self.alpha * torch.eye(J.shape[1]), error
        )
        psi = torch.bmm(J_T, u).squeeze(-1)
        psis = torch.split(psi, self.layer_sizes, dim=1)

        for i, layer in enumerate(self.layers):
            layer.r_prev = self.layers[i - 1].r if i != 0 else self.input
            grad_w = torch.bmm(psis[i].unsqueeze(2), layer.r_prev.unsqueeze(1)).mean(dim=0)
            grad_b = psis[i].mean(dim=0)

            layer.weights.data -= self.eta_ff * grad_w
            layer.bias.data -= self.eta_ff * grad_b
