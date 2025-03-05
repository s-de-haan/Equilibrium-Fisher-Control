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
            v_ff_current[i] = layer.v_ff
            v_current[i] = layer.v_ff
            r_current[i] = layer.r
            layer.activation_fn.reset_modulation()

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

                layer.activation_fn.set_modulation(e_psi_gamma)
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


class EFC_network_v2(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network_v2"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc

    @torch.no_grad()
    def _dynamical_inversion(self):
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        u_current = torch.zeros((self.bzs, self.output_size))
        u_int = torch.zeros((self.bzs, self.output_size))

        for _ in range(1, self.tmax):
            error = self._compute_error(self.layers[-1].r, self.targets)
            
            # Proportional and integral (PI) control
            u_int = u_int + self.dt * (error - self.alpha * u_current)
            u_next = u_int + self.k_p * error

            _, Js = self._calculate_full_jacobian()

            # Forward pass
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i-1].r if i != 0 else self.input
                layer.v_ff = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                psi = torch.bmm(u_next.unsqueeze(1), Js[i]).squeeze()
                gamma = self._compute_gamma(layer, i) if not self._first_task else 0.0
                e_psi_gamma = torch.tanh(psi + gamma) + 1

                layer.r = layer.r + self.tau * (e_psi_gamma * layer.r_ff - layer.r)
                layer.activation_fn.set_modulation(layer.r / (layer.r_ff + 1e-8))

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all():
                break
            u_current = u_next


class EFC_network_v3(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network_v3"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.fisher_normalization = config.fisher_normalization
        print("IM PRINTING THE FISHER NORMALIZATION", self.fisher_normalization)

    @torch.no_grad()
    def _dynamical_inversion(self):
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        u_current = torch.zeros((self.bzs, self.output_size))
        u_int = torch.zeros((self.bzs, self.output_size))

        for t in range(1, self.tmax):
            error = self._compute_error(self.layers[-1].r, self.targets)
            
            # Proportional and integral (PI) control
            u_int = u_int + self.dt * (error - self.alpha * u_current)
            u_next = u_int + self.k_p * error

            psis = self._calculate_psis(u_next)

            # Forward pass
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i-1].r if i != 0 else self.input
                layer.v_ff = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                psi = psis[i]
                gamma = self._compute_gamma(layer, i, self.fisher_normalization) if not self._first_task else 0.0
                e_psi_gamma = torch.tanh(psi + gamma) + 1

                layer.r = layer.r + self.dt / self.time_constant_ratio * (e_psi_gamma * layer.r_ff - layer.r)
                layer.activation_fn.set_modulation(layer.r / (layer.r_ff + 1e-8))

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all():
                break
            u_current = u_next
    
        
        
    @torch.no_grad()
    def _calculate_psis(self, u_next):
        L = len(self.layers)
        psi_list = [None] * L

        # Derivatives per layer
        activations_derivatives = [layer.activation_derivative(layer.v_ff) for layer in self.layers]
        
        # Last layer
        psi = u_next * activations_derivatives[-1]
        psi_list[-1] = psi
        
        # Backward from second-to-last to first
        for i in range(L - 2, -1, -1):
            psi = (psi @ self.layers[i + 1].weights) * activations_derivatives[i]
            psi_list[i] = psi
        
        return psi_list
        

class EFC_BP_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_BP_network"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.psi_lr = config.psi_lr
        self.inner_loss_fn = nn.CrossEntropyLoss(reduction='sum')
        self.alpha_psi = config.alpha_psi
        
    def _dynamical_inversion(self):
        # Initialize psi_params for each layer
        psi_params = [torch.zeros((self.bzs, layer.weights.shape[0]), requires_grad=True) for layer in self.layers]
        optimizer = torch.optim.SGD(psi_params, lr=self.psi_lr)

        # Convergence tracking
        converged_mask = torch.zeros(self.bzs, dtype=torch.bool)
        prev_psi_l2 = torch.ones(self.bzs)

        # Iterative optimization
        for t in range(1, self.tmax):
            # Reset optimizers
            optimizer.zero_grad()

            # Forward pass: Compute activations layer by layer
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i-1].r if i != 0 else self.input

                layer.r_ff = layer.activation_fn(layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0))  # Nonlinear activation
                
                psi = psi_params[i]
                e_psi_gamma = torch.tanh(psi) + 1
                
                layer.r = e_psi_gamma * layer.r_ff

            # Compute loss at the output and Backpropagate and update gradients
            psi_l2 = torch.sum(torch.stack([torch.sum(psi ** 2, dim=1) for psi in psi_params]), dim=0)
            task_loss = self.inner_loss_fn(self.layers[-1].r, self.targets)
            loss = task_loss + 0.5 * self.alpha_psi * psi_l2.sum()

            loss.backward(retain_graph=True) # TODO does gamma need a graph?
            optimizer.step()

            # print(torch.max(psi_params[-1]), torch.max(psi_params[-1].grad))

            # Check convergence
            norm_diff = torch.abs(psi_l2 - prev_psi_l2)
            converged_mask |= norm_diff < self.eps
            prev_psi_l2 = psi_l2

            if converged_mask.all():
                break
            
class EFC_CNN_network(EFC_CNN_network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_CNN_network"):
        EFC_CNN_network.__init__(self, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.psi_lr = config.psi_lr
        self.inner_loss_fn = nn.CrossEntropyLoss(reduction='sum')

    def _dynamical_inversion(self):
        """
        Perform dynamical inversion to update activations with modulation.
        Adapted for convolutional layers with feature map handling.
        """
        # Initialize activations from layer outputs
        v_ff_current = [layer.v_ff.detach().clone() for layer in self.layers]
        v_current = [layer.v_ff.detach().clone() for layer in self.layers]
        r_current = [layer.r.detach().clone() for layer in self.layers]

        # Initialize psi parameters for each layer
        psi_params = []
        for layer in self.layers:
            # Shape matches the layer's linear_activations (e.g., [batch_size, channels, height, width])
            psi = torch.zeros_like(layer.v_ff, requires_grad=True)
            psi_params.append(psi)

        # Initialize gamma (Fisher modulation) for each layer
        gammas = []
        for i, layer in enumerate(self.layers):
            gamma = self._compute_fisher_modulation_conv(layer, i) if not self._first_task else torch.zeros_like(layer.v_ff)
            gammas.append(gamma)

        # Set up optimizer and convergence tracking
        optimizer = torch.optim.SGD(psi_params, lr=self.psi_lr)
        converged_mask = torch.zeros(self.bzs, dtype=torch.bool)
        psi_prev = [psi.detach().clone() for psi in psi_params]

        for _ in range(self.tmax - 1):
            if converged_mask.all():
                break

            optimizer.zero_grad()

            # Detach tensors to prevent unwanted gradient tracking
            for i in range(len(v_current)):
                v_current[i] = v_current[i].detach()
                r_current[i] = r_current[i].detach()
                v_ff_current[i] = v_ff_current[i].detach()

            for i, layer in enumerate(self.layers):
                r_prev = r_current[i-1] if i > 0 else self.input
                
                # Compute feedforward pass based on layer type
                if isinstance(layer, EFC_Conv_layer):
                    v_ff = layer.conv(r_prev)
                else:  # Fully connected layer
                    r_prev_flat = r_prev.view(self.bzs, -1)
                    v_ff = r_prev_flat.mm(layer.weight.t()) + layer.bias.unsqueeze(0)
                
                # Apply modulation
                psi = psi_params[i]
                gamma = gammas[i].expand_as(psi) if i >= 4 else gammas[i].view(1, -1, 1, 1).expand_as(psi)
                e_psi_gamma = torch.tanh(psi + gamma) + 1
                r_current[i] = e_psi_gamma * layer.activation_fn(v_current[i])
                v_ff_current[i] = v_ff
                
                # Update x for next layer with post-processing
                x = layer.activation_fn(v_ff)  # Apply ReLU (or Identity for FC)
                if i < 4:  # Convolutional layers only
                    x = self.bn_layers[i](x)   # Batch normalization
                    x = self.pool(x)           # Max-pooling

            # Compute loss (assumes last layer is fully connected for classification)
            loss = self.inner_loss_fn(r_current[-1], self.targets)
            
            # Backpropagate and update psi
            loss.backward(retain_graph=True)
            optimizer.step()

            # Check convergence per batch element
            psi_changes = []
            for i, psi in enumerate(psi_params):
                # Norm over spatial/channel dims for conv, or neuron dim for FC
                if len(psi.shape) == 4:  # Convolutional layer
                    change = torch.norm(psi - psi_prev[i], dim=(1, 2, 3))
                else:  # Fully connected layer
                    change = torch.norm(psi - psi_prev[i], dim=1)
                psi_changes.append(change)
            max_psi_change = torch.stack(psi_changes, dim=0).max(dim=0).values  # [batch_size]
            converged_mask |= max_psi_change < self.eps
            psi_prev = [psi.detach().clone() for psi in psi_params]

        # Store final steady-state values
        rs = [self.input]
        for i, layer in enumerate(self.layers):
            layer.r = r_current[i].detach().clone()
            layer.r_ff = layer.activation_fn(v_ff_current[i]).detach().clone()
            layer.r_prev = rs[i]
            rs.append(layer.r)