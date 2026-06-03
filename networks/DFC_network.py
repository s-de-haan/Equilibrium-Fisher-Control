import torch
import math

from networks.network_interface import *
from networks.layers import DFC_layer
from networks.activation_function import *

check_nan = lambda x: torch.isnan(x).any().item()

class DFC_network(Network, JacobianInterface):
    def __init__(self, config, name="DFC_network") -> None:
        if "activation_fun" in config:
            if config["activation_fun"]=="Tanh":
                act_fun = Tanh
            else:
                act_fun = ReLU
        else:
            act_fun = ReLU
        Network.__init__(self, DFC_layer, act_fun, Linear, config, name)
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
            if converged_mask.float().mean().item() >= 1:
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
                v_ff_current[i] += (r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0) - v_ff_current[i])
                v_ff_current[i] = v_ff_current[i].clone().detach()
                v_fb_current[i] = torch.bmm(Js[i].transpose(1, 2), u_next.unsqueeze(2)).squeeze(2)
                
                # Soma with apical
                tau = self.dt / self.time_constant_ratio
                v_current[i] += tau * (v_fb_current[i] + v_ff_current[i] - v_current[i]) * (~converged_mask.unsqueeze(1))
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


class DFC_Mult_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="DFC_Mult_network"):
        Network.__init__(self, DFC_layer, ReLU, Linear, config, name)
        JacobianInterface.__init__(self, config)

    @torch.no_grad()
    def _dynamical_inversion(self):
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        u_current = torch.zeros((self.bzs, self.output_size))

        for _ in range(1, self.tmax):
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



class DFCL_network(Network, JacobianInterface):

    def __init__(self, config, name="DFCL_network"):
        # Choose activation function
        if "activation_fun" in config:
            if config["activation_fun"] == "Tanh":
                act_fun = Tanh
            else:
                act_fun = ReLU
        else:
            act_fun = ReLU

        # ----- Initialize network layers -----
        Network.__init__(self, DFC_layer, act_fun, Linear, config, name)

        # ----- Initialize inversion interface -----
        JacobianInterface.__init__(self, config)

        # ----- Add lateral + feedback connections -----
        self._init_feedback_connections()
        self.converged_per_batch = []
        self.mean_convergence_per_epoch = []

    # ---------------------------------------------------
    # Feedback connections: layer i+1 → layer i
    # ---------------------------------------------------
    def _init_feedback_connections(self):
        """
        feedback_weights[i] maps: layer_(i+1) → layer_(i).
        
        Shapes:
            feedback[i] = [size_i , size_(i+1)]
        Where size_k is layer out_features of layer k.
        
        For i = 0 (first hidden layer),
            feedback[0] = [hidden1 , input_dim]
        """

        self.feedback_layers = nn.ModuleList()

        # Gather all sizes: input + all layers' output sizes
        all_sizes = [self.layers[0].in_features] + self.layer_sizes

        # Add a feedback matrix for each layer 
        for i in range(len(self.layers)):
            in_dim = all_sizes[i]       # lower layer dimension
            out_dim = all_sizes[i-1]  # higher layer dimension

            # Weight matrix: (higher_layer ← lower_layer)
            if in_dim==784:
                W_fb = Layer(out_dim, in_dim, activation_fn=Linear(), name="Linear", use_bias=False)
            else:
                W_fb = Layer(out_dim, in_dim, activation_fn=Linear(), name="Linear", use_bias=False)
            k = 1.0 / in_dim
            bound = math.sqrt(k)

            nn.init.uniform_(W_fb.weights, -bound, bound)

            if W_fb.bias is not None:
                nn.init.uniform_(W_fb.bias, -bound, bound)

            self.feedback_layers.append(W_fb)

        print("[DFCL] Feedback connections initialized.")


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
        for t_step in range(self.tmax - 1):
            # Stop if converged

            error = self._compute_error(r_current[-1], self.targets)
            
            # Proportional and integral (PI) control.
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Compute convergence check
            #converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            converged_mask |= (torch.abs(error)).mean(dim=1) < 0.00001

            if converged_mask.float().mean().item() >= 0.95:
                self.converged_per_batch.append(t_step)
                break

            _, Js = self._calculate_full_jacobian()
            
            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                r_prev = r_current[i - 1] if i != 0 else self.input

                # Soma with apical
                tau = self.dt / self.time_constant_ratio

                # Basal and apical
                v_ff_current[i] += (r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0) - v_ff_current[i]) * (~converged_mask.unsqueeze(1))
                v_ff_current[i] = v_ff_current[i].clone().detach()
                
                if i!=len(self.layers)-1:
                    v_fb_current[i] = self.feedback_layers[-(i+1)](u_next) * layer.activation_derivative(v_ff_current[i])
                    v_current[i] += tau * (v_fb_current[i] + v_ff_current[i] - v_current[i]) * (~converged_mask.unsqueeze(1))
                else:
                    v_fb_current[i] = torch.bmm(Js[i].transpose(1, 2), u_next.unsqueeze(2)).squeeze(2)
                    v_current[i] += tau * (v_fb_current[i] + v_ff_current[i] - v_current[i]) * (~converged_mask.unsqueeze(1))

                
                r_current[i] = layer.activation_fn(v_current[i])

                layer.v_ff = v_current[i]
                layer.r = r_current[i]

            
            u_int_current = u_int_next
            u_current = u_next
            

        if not converged_mask.float().mean().item() >= 0.95:
                self.converged_per_batch.append(self.tmax)
        # Steady-state values per layer
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i]) 
            layer.r_prev = rs[i]
            rs.append(r_current[i])

    
    @torch.no_grad()
    def generate_samples(self, target_activity, init_input, tmax=500, threshold=0.01):

        # === Setup ===
        self.bzs = target_activity.shape[0]

        #init_input_scaled = init_input.to(self.device)*10 - (5)
        #input_pre = init_input_scaled
        input_pre = init_input
        self.input = init_input#torch.sigmoid(input_pre)

        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        v_fb_current = [torch.zeros((self.bzs, lod), device=self.device) for lod in layer_out_dims]
        v_ff_current = [torch.zeros((self.bzs, lod), device=self.device) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod), device=self.device) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod), device=self.device) for lod in layer_out_dims]
        u_current = torch.zeros((self.bzs, self.layer_sizes[-1]), device=self.device)
        u_int_current = torch.zeros_like(u_current)

        u_current2 = torch.zeros((self.bzs, self.layer_sizes[-2]), device=self.device)
        u_int_current2 = torch.zeros_like(u_current2)

        # Forward init
        with torch.no_grad():
            r_prev = self.input
            for i, layer in enumerate(self.layers):
                v_ff_current[i] = r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                v_current[i] = v_ff_current[i]
                r_current[i] = layer.activation_fn(v_ff_current[i])
                r_prev = r_current[i]
        _ = self(self.input)


        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool, device=self.device)

        diffs = []

        # === Dynamics ===
        for t in range(tmax - 1):
            if converged_mask.all():
                break
            
            error = self._compute_error(r_current[-1], target_activity)

            # PI control
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            converged_mask = (torch.abs(error)).mean(dim=1) < threshold ### careful here

            _, Js = self._calculate_full_jacobian()

            for i, layer in enumerate(self.layers):
                r_prev = r_current[i - 1] if i != 0 else self.input

                # Basal + apical feedback
                v_ff_current[i] += (r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0) - v_ff_current[i]) 
                v_ff_current[i] = v_ff_current[i].clone().detach()
                

                tau = self.dt / self.time_constant_ratio

                if i!=1:
                    v_fb_current[i] = (self.feedback_layers[i+1](u_next)) * layer.activation_derivative(v_current[i])
                    v_current[i] += tau * (v_fb_current[i]* (~converged_mask.unsqueeze(1)) + v_ff_current[i] - v_current[i])
                    
                else:
                    v_current[i] +=  tau * (v_ff_current[i] - v_current[i]) * (~converged_mask.unsqueeze(1))
                    #v_fb_current[i] = torch.bmm(Js[i].transpose(1, 2),u_next.unsqueeze(2)).squeeze(2)
                    #v_current[i] += tau * (v_fb_current[i] + v_ff_current[i] - v_current[i]) * (~converged_mask.unsqueeze(1))

                r_current[i] = layer.activation_fn(v_current[i])

                layer.v_ff = v_current[i]
                layer.r = r_current[i]

            error2 = self._compute_error_mse(self.layers[0].activation_fn(v_ff_current[0]), r_current[-2])

            # PI control
            u_int_next2 = u_int_current2 + self.dt * (error2 - self.alpha * u_current2)
            u_next2 = u_int_next2 + self.k_p * error2

            u_int_current = u_int_next
            u_current = u_next

            u_int_current2 = u_int_next2
            u_current2 = u_next2

            
            grad_input = (self.feedback_layers[0](r_current[-2]))
            grad_input = torch.clamp(grad_input, -0.4242 ,2.8215)
            grad_input += (self.feedback_layers[0](u_next2))
            
            #grad_input = (self.feedback_layers[0](r_current[-2]))
            #input_pre += (grad_input + init_input_scaled - input_pre) * ~converged_mask.unsqueeze(1)
            #input_pre += tau * (grad_input)
            input_pre += tau* (grad_input - self.input)
            input_post = torch.clamp(self.feedback_layers[0].activation_fn(input_pre), -0.4242 ,2.8215)
            self.input = input_post
            diff = torch.abs(r_current[0] - self.layers[0](self.input)).mean().detach().cpu().numpy()
            diffs.append(diff)
            
        # === Collect steady-state representations ===
        rs = [self.input]
        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])

        generated_inputs = self.input#torch.clamp(self.feedback_layers[0](r_current[0]), -0.4242 ,2.8215).clone().detach()

        return generated_inputs, torch.softmax(rs[-1],dim=1).clone().detach(), diffs
    


class DFCL_network_multilayer(Network, JacobianInterface):

    def __init__(self, config, name="DFCL_network"):
        # Choose activation function
        if "activation_fun" in config:
            if config["activation_fun"] == "Tanh":
                act_fun = Tanh
            else:
                act_fun = ReLU
        else:
            act_fun = ReLU

        # ----- Initialize network layers -----
        Network.__init__(self, DFC_layer, act_fun, Linear, config, name)

        # ----- Initialize inversion interface -----
        JacobianInterface.__init__(self, config)

        # ----- Add lateral + feedback connections -----
        self._init_feedback_connections()
        self.converged_per_batch = []
        self.mean_convergence_per_epoch = []

    # ---------------------------------------------------
    # Feedback connections: layer i+1 → layer i
    # ---------------------------------------------------
    def _init_feedback_connections(self):
        """
        feedback_weights[i] maps: layer_(i+1) → layer_(i).
        
        Shapes:
            feedback[i] = [size_i , size_(i+1)]
        Where size_k is layer out_features of layer k.
        
        For i = 0 (first hidden layer),
            feedback[0] = [hidden1 , input_dim]
        """

        self.feedback_layers = nn.ModuleList()

        # Gather all sizes: input + all layers' output sizes
        all_sizes = [self.layers[0].in_features] + self.layer_sizes

        # Add a feedback matrix for each layer 
        for i in range(len(self.layers)):
            in_dim = all_sizes[i]       # lower layer dimension
            out_dim = all_sizes[i+1]  # higher layer dimension

            # Weight matrix: (higher_layer ← lower_layer)
            if in_dim==784:
                W_fb = Layer(out_dim, in_dim, activation_fn=Linear(), name="Linear", use_bias=True)
            else:
                W_fb = Layer(out_dim, in_dim, activation_fn=Linear(), name="Linear", use_bias=False)
            k = 1.0 / in_dim
            bound = math.sqrt(k)

            nn.init.uniform_(W_fb.weights, -bound, bound)

            if W_fb.bias is not None:
                nn.init.uniform_(W_fb.bias, -bound, bound)

            self.feedback_layers.append(W_fb)

        print("[DFCL] Feedback connections initialized.")


    @torch.no_grad()
    def _dynamical_inversion(self):
        # Setup
        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]
        n_layers = len(self.layers)
        thresholds = [0.00001,0.00001,0.00001,0.00001]

        v_fb_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]
        v_ff_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]

        # One PI controller per layer
        u_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]
        u_int_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]

        for i, layer in enumerate(self.layers):
            v_ff_current[i] = layer.v_ff
            v_current[i] = layer.v_ff
            r_current[i] = layer.r

        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool, device=self.input.device)
        tau = self.dt / self.time_constant_ratio

        # Simulate tmax timesteps
        for t_step in range(self.tmax - 1):

            # Convergence check from current output activity
            output_error = self._compute_error(r_current[-1], self.targets)
            #converged_mask |= (torch.abs(output_error)).mean(dim=1) < 0.00001
            converged_mask_per_layer = [torch.zeros((self.bzs,), dtype=torch.bool, device=self.input.device)] * n_layers


            if converged_mask.float().mean().item() >= 1:
                self.converged_per_batch.append(t_step)
                break

            _, Js = self._calculate_full_jacobian()

            # Hold next controller values for this sweep
            u_int_next = [u.clone() for u in u_int_current]
            u_next = [u.clone() for u in u_current]

            # Top-down sweep
            for i in reversed(range(n_layers)):
                layer = self.layers[i]
                r_prev = r_current[i - 1] if i != 0 else self.input

                # Basal update for this layer
                v_ff_current[i] += (
                    r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0) - v_ff_current[i]
                ) #* (~converged_mask.unsqueeze(1))
                v_ff_current[i] = v_ff_current[i].clone().detach()

                if i == n_layers - 1:
                    # =================================================
                    # Output layer: controller first, then activity
                    # =================================================
                    error_i = self._compute_error(r_current[i], self.targets)

                    u_int_next[i] = u_int_current[i] + self.dt * (
                        error_i - self.alpha * u_current[i]
                    )
                    u_next[i] = u_int_next[i] + self.k_p * error_i

                    v_fb_current[i] = torch.bmm(
                        Js[i].transpose(1, 2),
                        u_next[i].unsqueeze(2)
                    ).squeeze(2)

                    v_current[i] += tau * (
                        v_fb_current[i] + v_ff_current[i] - v_current[i]
                    ) #* (~converged_mask.unsqueeze(1))

                    r_current[i] = layer.activation_fn(v_current[i])

                    layer.v_ff = v_current[i]
                    layer.r = r_current[i]

                else:
                    # =================================================
                    # Lower layers: activity first, then controller
                    # =================================================
                    v_fb_current[i] = (
                        self.feedback_layers[i + 1](u_next[i + 1])
                        * layer.activation_derivative(v_ff_current[i])
                    ) * (~converged_mask_per_layer[i + 1].unsqueeze(1))
                    """v_fb_current[i] = (
                        u_next[i + 1]@self.layers[i + 1].weights
                        * layer.activation_derivative(v_ff_current[i])
                    )"""

                    v_current[i] += tau * (
                        v_fb_current[i] + v_ff_current[i] - v_current[i] - u_next[i]* (~converged_mask_per_layer[i].unsqueeze(1))
                     ) #* (~converged_mask.unsqueeze(1))

                    r_current[i] = layer.activation_fn(v_current[i])

                    layer.v_ff = v_current[i]
                    layer.r = r_current[i]

                    error_i = self._compute_error_mse(
                        layer.activation_fn(v_ff_current[i]),
                        r_current[i]
                    )

                    u_int_next[i] = u_int_current[i] + self.dt * (
                        error_i - self.alpha * u_current[i]
                    )
                    u_next[i] = u_int_next[i] + self.k_p * error_i

                converged_mask_per_layer[i] = torch.abs(error_i).mean(dim=1) < thresholds[i]
            converged_mask |= torch.norm(u_next[-1] - u_current[-1], dim=1) < self.eps

            # Commit controller updates after full sweep
            u_int_current = u_int_next
            u_current = u_next

        if not converged_mask.float().mean().item() >= 0.95:
            self.converged_per_batch.append(self.tmax)

        # Steady-state values per layer
        rs = [self.input]

        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])

    @torch.no_grad()
    def generate_samples(self, target_activity, init_input=None, batch_size=10, tmax=500, pixel_std=None, pixel_mean=None, thresholds=[0.1, 0.1, 0.001]):
        # === Setup ===
        self.targets = target_activity
        self.bzs = target_activity.shape[0]

        # Initialize input (random if none provided)
        if init_input is None:
            init_input_scaled = (
                torch.clamp(torch.randn(batch_size, 28 * 28, device=self.device)
                * pixel_std.flatten().to(self.device)
                + pixel_mean.flatten().to(self.device), -0.4242, 2.8215)
            )
            self.input = init_input_scaled
        else:
            self.input = init_input.to(self.device)

        layer_out_dims = [layer.weights.shape[0] for layer in self.layers]

        v_fb_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]
        v_ff_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]
        v_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]
        r_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]

        # One PI controller per layer
        u_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]
        u_int_current = [torch.zeros((self.bzs, lod), device=self.input.device) for lod in layer_out_dims]

        # Forward init
        r_prev = self.input
        for i, layer in enumerate(self.layers):
            v_ff_current[i] = r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
            v_current[i] = v_ff_current[i].clone()
            r_current[i] = layer.activation_fn(v_ff_current[i])
            r_prev = r_current[i]

        _ = self(self.input)

        diffs = []
        diffs2 = []
        diffs3 = []
        video = [self.input[12].clone()]
        rs_hist = [self.layers[-1].r[12].clone()]

        tau = self.dt / self.time_constant_ratio
        n_layers = len(self.layers)

        # Simulate tmax timesteps
        for t_step in range(tmax - 1):
            # store updated controllers for this step
            u_next = [u.clone() for u in u_current]
            u_int_next = [u.clone() for u in u_int_current]

            # convergence mask for every controller
            converged_mask_per_layer = [None] * n_layers

            # Iterate from output layer down to first hidden layer
            for i in reversed(range(n_layers)):
                layer = self.layers[i]

                # --- 1) Update controller for this layer ---
                if i != n_layers - 1:
                    error_i = self._compute_error_mse(
                        layer.activation_fn(v_ff_current[i]),
                        r_current[i]
                    )
                else:
                    error_i = self._compute_error(r_current[i], self.targets)

                u_int_next[i] = u_int_current[i] + self.dt * (error_i - self.alpha * u_current[i])
                u_next[i] = u_int_next[i] + self.k_p * error_i 

                converged_mask_per_layer[i] = torch.abs(error_i).mean(dim=1) < thresholds[i]

                # --- 2) Update activity for this layer ---
                r_prev = r_current[i - 1] if i != 0 else self.input

                # Basal update
                v_ff_current[i] += (
                    r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0) - v_ff_current[i]
                )
                v_ff_current[i] = v_ff_current[i].clone().detach()

                # Apical / soma update
                if i != n_layers - 1:
                    v_fb_current[i] = (self.feedback_layers[i + 1](u_next[i + 1])) #*  layer.activation_derivative(v_current[i])
                    #v_fb_current[i] = (u_next[i + 1])@self.layers[i+1].weights
                    v_current[i] += tau * (
                        v_fb_current[i] * (~converged_mask_per_layer[i + 1].unsqueeze(1))
                        + v_ff_current[i] - v_current[i]
                        - u_next[i] * (~converged_mask_per_layer[i].unsqueeze(1))
                    )
                else:
                    v_current[i] += tau * (v_ff_current[i] - v_current[i])

                r_current[i] = layer.activation_fn(v_current[i])

                layer.v_ff = v_current[i]
                layer.r = r_current[i]

            # Commit controller updates after full downward sweep
            u_int_current = u_int_next
            u_current = u_next

            # Input update
            grad_input = self.feedback_layers[0](self.layers[0].activation_fn(v_current[0])) 
            grad_input += self.feedback_layers[0](u_next[0]) 

            self.input += tau * (grad_input - torch.abs(self.input))
            self.input = torch.clamp(self.input, -0.4242, 2.8215)

            video.append(self.input[12].clone())
            rs_hist.append(self.layers[-1].r[12].clone())

            diff = torch.abs(
                r_current[0] - self.layers[0].activation_fn(v_ff_current[0])
            ).mean().detach().cpu().numpy()
            diffs.append(diff)

            if len(self.layers) > 1:
                diff2 = torch.abs(
                    r_current[1] - self.layers[1].activation_fn(v_ff_current[1])
                ).mean().detach().cpu().numpy()
                diffs2.append(diff2)
            
            if len(self.layers) > 2:
                diff3 = torch.abs(
                    r_current[2] - self.layers[2].activation_fn(v_ff_current[2])
                ).mean().detach().cpu().numpy()
                diffs3.append(diff3)


        # === Collect steady-state representations ===
        rs = [self.input]
        for i, layer in enumerate(self.layers):
            layer.r = r_current[i]
            layer.r_ff = layer.activation_fn(v_ff_current[i])
            layer.r_prev = rs[i]
            rs.append(r_current[i])

        print(converged_mask_per_layer[-1].sum())

        generated_inputs = self.input

        if len(diffs) > 0:
            print(diffs[-1])
        if len(diffs2) > 0:
            print(diffs2[-1])
        if len(diffs3) > 0:
            print(diffs3[-1])

        return generated_inputs, rs, video, diffs, diffs2, diffs3, rs_hist