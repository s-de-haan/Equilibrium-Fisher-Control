import torch

from networks.network_interface import *
from networks.layers import *
from networks.activation_function import *

"""
    Original implementation adapted from the DFC_Mult_network with gamma term
"""
class EFC_network(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.clamp = config.clamp

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

            psis = self._calculate_psis(u_next)

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                layer.r_prev = r_current[i - 1] if i != 0 else self.input

                # Basal
                v_ff_current[i] = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)

                # Apical with teaching signal and Fisher modulation
                psi = psis[i] # = torch.bmm(u_next.unsqueeze(1), Js[i]).squeeze()
                gamma = gammas[i]

                e_psi_gamma = torch.tanh(psi + gamma) + 1

                # Soma with modulation
                # tau = layer.tau # self.dt / self.time_constant_ratio
                v_current[i] += self.tau * (e_psi_gamma * v_ff_current[i] - v_current[i])

                # layer.activation_fn.set_modulation(e_psi_gamma)
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
    Greatly simplified version with correct derivatives
"""
class EFC_network_v2(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network_v2"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta

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


"""
    Efficient implementation with implicit Jacobian calculation for scalability
"""
class EFC_network_v3(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network_v3"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.fisher_normalization = config.fisher_normalization

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


class EFC_network_v4(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network_v4"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)

        self.beta = config.beta_efc

    @torch.no_grad()
    def _dynamical_inversion(self):
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)
        u_current = torch.zeros((self.bzs, self.output_size))

        for t in range(1, self.tmax):
            error = self._compute_error(self.layers[-1].r, self.targets)
            
            # Proportional and integral (PI) control
            u_next = self.k_p * error
            
            psis = self._calculate_psis(u_next)

            # Forward pass
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i-1].r if i != 0 else self.input
                layer.v_ff = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)
                
                psi = psis[i] # ψ = (J^T u) ⊙ r_ff = diag(r_ff) J^T u
                # gamma = self._compute_gamma(layer, i)
                # if not self._first_task:
                #     gamma = torch.clamp(gamma, min=-torch.abs(psi), max=torch.abs(psi))
                e_psi_gamma = torch.tanh(psi) + 1 # + gamma) + 1

                layer.r = layer.r + self.tau * (e_psi_gamma * layer.r_ff - layer.r)

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all():
                break
            u_current = u_next
        # err
        # if self._first_task:
        #     print(t, torch.min(psi).item(), torch.max(psi).item())
        # else:
        #     print(t, torch.min(psi).item(), torch.max(psi).item(), torch.max(gamma).item(), torch.max(gamma).item())
        # print("error: ",[f"{x:.4f}" for x in error[0].tolist()])
        # err


class EFC_network_v5(Network, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_network"):
        Network.__init__(self, DFC_layer, Softplus, Softplus, config, name)
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)
        
        self.beta = config.beta_efc
        self.tau = config.taus

    @torch.no_grad()
    def _dynamical_inversion(self):
        u_current = torch.zeros((self.bzs, self.output_size))
        u_int_current = torch.zeros((self.bzs, self.output_size))
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool)

        for t in range(self.tmax):
            error = self._compute_error(self.layers[-1].r, self.targets)
            
            # Proportional and integral (PI) control
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Compute convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all():
                break

            psis = self._calculate_psis(u_next)

            # Iterate over layers with control signal
            for i, layer in enumerate(self.layers):
                layer.r_prev = self.layers[i-1].r if i != 0 else self.input

                # Basal
                layer.v_ff = layer.r_prev.mm(layer.weights.t()) + layer.bias.unsqueeze(0)
                layer.r_ff = layer.activation_fn(layer.v_ff)

                # Apical with teaching signal and Fisher modulation
                psi = psis[i]
                e_psi_gamma = torch.tanh(psi) + 1

                # Soma with modulation
                tau = self.tau[i]
                layer.r = layer.r + tau * (e_psi_gamma * layer.r_ff - layer.r)

            u_int_current = u_int_next
            u_current = u_next


class EFC_Conv_v5_network(nn.Module, JacobianInterface, FisherInterface):
    def __init__(self, config, name="EFC_Conv_v5_network"):
        """Initialize the convolutional EFC network."""
        super().__init__()
        JacobianInterface.__init__(self, config)
        FisherInterface.__init__(self)

        self.loss_fn = nn.MSELoss() if config.loss_fn == "mse" else nn.CrossEntropyLoss()
        self.loss_fn_name = config.loss_fn
        self.device = config.device
        self.lr = config.lr
        self.tau = config.taus
        self.name = name
        
        # Override layer creation
        self.create_network(config)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.pool_indices = [None] * 4  # To store indices for each pooling layer
        
        # Batch normalization layers (not modulated)
        self.bn_layers = nn.ModuleList()
        for _ in range(4):  # 4 convolutional layers
            self.bn_layers.append(nn.BatchNorm2d(64))

    def create_network(self, config):
        """Create the network: 4 conv layers + 1 FC layer."""
        self.layers = nn.ModuleList()
        
        # Input channels from config (e.g., 1 for grayscale, 3 for RGB)
        current_channels = config.in_channels
        
        # 4 Convolutional layers
        for _ in range(4):
            conv_layer = DFC_Conv_layer(
                in_channels=current_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                activation_fn=Softplus()
            )
            self.layers.append(conv_layer)
            current_channels = 64
        
        # Fully connected output layer (in_features set dynamically in forward)
        self.layers.append(
            DFC_layer(
                in_features=64,  # TODO Placeholder, adjusted later
                out_features=config.num_classes,
                activation_fn=Softplus() 
            )
        )

    def calculate_loss(self, y_hat, y):
        self.loss = self.loss_fn(y_hat, y)
        return self.loss

    def forward(self, x):
        """Forward pass through the network."""
        self.input = x
        self.bzs = x.shape[0]
        
        # Process 4 convolutional modules
        for i in range(4):
            conv_layer = self.layers[i]
            bn_layer = self.bn_layers[i]
            x = conv_layer.conv(x)              # Raw convolution output (v_ff)
            conv_layer.v_ff = x                 # Store pre-activation
            x = conv_layer.activation_fn(x)     # Apply activation
            conv_layer.r = x                    # Initialize r with feedforward output
            x = bn_layer(x)                     # Batch normalization
            x, indices = self.pool(x)           # Max pooling with indices
            self.pool_indices[i] = indices      # Store for feedback
            conv_layer.r_out = x                # Store post-pooling output
        
        # Flatten and apply FC layer
        x = x.view(self.bzs, -1)
        fc_layer = self.layers[-1]
        if fc_layer.in_features != x.shape[1]:
            fc_layer.in_features = x.shape[1]
            fc_layer._create_init_layer()
        
        x = fc_layer.feedforward[0](x)          # Linear transformation (v_ff)
        fc_layer.v_ff = x                       # Store pre-activation
        x = fc_layer.activation_fn(x)           # Apply activation (identity here)
        fc_layer.r = x                          # Initialize r
        self.y_hat = x

        return x

    @torch.no_grad()
    def _dynamical_inversion(self):
        """Override dynamical inversion to handle conv and FC layers."""
        u_current = torch.zeros((self.bzs, self.output_size), device=self.device)
        u_int_current = torch.zeros((self.bzs, self.output_size), device=self.device)
        converged_mask = torch.zeros((self.bzs,), dtype=torch.bool, device=self.device)

        for t in range(self.tmax):
            for bn in self.bn_layers:
                bn.eval()  # Freeze BatchNorm stats
            error = self._compute_error(self.layers[-1].r, self.targets)
            
            # PI control
            u_int_next = u_int_current + self.dt * (error - self.alpha * u_current)
            u_next = u_int_next + self.k_p * error

            # Convergence check
            converged_mask |= torch.norm(u_next - u_current, dim=1) < self.eps
            if converged_mask.all():
                break

            psis = self._calculate_psis(u_next)

            # Update each layer
            for i, layer in enumerate(self.layers):
                # Set r_prev based on previous layer's output
                if i == 0:
                    layer.r_prev = self.input
                elif isinstance(self.layers[i-1], DFC_Conv_layer):
                    layer.r_prev = self.pool(self.layers[i-1].r)[0] # not the indices
                elif isinstance(self.layers[i-1], DFC_layer):
                    layer.r_prev = self.layers[i-1].r

                # Compute feedforward activations
                if isinstance(layer, DFC_Conv_layer):
                    layer.v_ff = F.conv2d(
                        layer.r_prev,
                        layer.weights,
                        bias=layer.bias,
                        stride=layer.stride,
                        padding=layer.padding
                    )
                    if i < len(self.bn_layers):  # Apply BN for conv layers
                        layer.v_ff = self.bn_layers[i](layer.v_ff)
                elif isinstance(layer, DFC_layer):
                    layer.v_ff = layer.r_prev.view(self.bzs, -1) @ layer.weights.t() + layer.bias.unsqueeze(0)
                
                layer.r_ff = layer.activation_fn(layer.v_ff)
                
                # Modulate activations with control signal
                psi = psis[i]
                e_psi_gamma = torch.tanh(psi) + 1
                tau = self.tau[i]
                layer.r = layer.r + tau * (e_psi_gamma * layer.r_ff - layer.r)

            u_int_current = u_int_next
            u_current = u_next

        print(t)

    @torch.no_grad()
    def _calculate_psis(self, u):
        """Calculate control signals (psis) for each layer in the feedback path."""
        L = len(self.layers)  # Total layers: 4 conv + 1 FC
        psi_list = [None] * L
        activations_derivatives = [layer.activation_derivative(layer.v_ff) for layer in self.layers]
        
        # Compute psi for the FC layer (last layer)
        psi = u * activations_derivatives[-1]  # [batch_size, num_classes]
        psi_list[-1] = psi
        
        # Transition from FC to last conv layer (index 3)
        psi = psi @ self.layers[-1].weights    # [batch_size, 64]
        psi = psi.view(self.layers[-2].r_out.shape)  # [batch_size, 64, 1, 1]
        
        # Unpool Layer 3's output from [1, 1] to [3, 3]
        psi = F.max_unpool2d(
            psi,
            self.pool_indices[3],
            kernel_size=2,
            stride=2,
            output_size=self.layers[3].r.shape[2:]  # [3, 3]
        )
        psi_list[3] = psi  # [batch_size, 64, 3, 3]
        
        # Propagate psi backward through conv layers 2, 1, 0
        for i in range(2, -1, -1):
            psi = F.max_unpool2d(
                psi,
                self.pool_indices[i],
                kernel_size=2,
                stride=2,
                output_size=self.layers[i].r.shape[2:]  # Spatial size before pooling
            )
            psi = F.conv_transpose2d(
                psi,
                self.layers[i + 1].weights,
                stride=self.layers[i + 1].stride,
                padding=self.layers[i + 1].padding
            )
            psi = psi * activations_derivatives[i]
            psi_list[i] = psi
        
        return psi_list