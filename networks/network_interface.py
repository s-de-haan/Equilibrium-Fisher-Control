import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import *

class Network(nn.Module):
    def __init__(self, layer_class, activation_fn, out_activation_fn, config, name):
        super().__init__()
        
        self.create_network(layer_class, activation_fn, out_activation_fn, config)
        self.loss_fn = nn.MSELoss() if config.loss_fn == "mse" else nn.CrossEntropyLoss()
        self.loss_fn_name = config.loss_fn
        self.device = config.device
        self.lr = config.lr
        self.name = name

    @property
    def layer_sizes(self):
        return [layer.out_features for layer in self.layers]

    @property
    def activations(self):
        return [layer.r for layer in self.layers]

    @property
    def linear_activations(self):
        return [layer.v_ff for layer in self.layers]

    def forward(self, x):
        self.input = x
        self.bzs = x.shape[0]
        for layer in self.layers:
            x = layer(x)
        self.y_hat = x
        return x

    def create_network(self, layer_class, activation_fn, out_activation_fn, config):
        _layers = config.layers
        self.layers = nn.ModuleList()
        for i in range(len(_layers) - 2):
            self.layers.append(
                layer_class(
                    _layers[i],
                    _layers[i + 1],
                    activation_fn=activation_fn(),
                )
            )
        self.layers.append(
            layer_class(
                _layers[-2],
                _layers[-1],
                activation_fn=out_activation_fn(),
            )
        )

    def calculate_loss(self, y_hat, y):
        self.loss = self.loss_fn(y_hat, y)
        return self.loss

class EFC_CNN_network(nn.Module):
    def __init__(self, activation_fn, out_activation_fn, config, name="EFC_CNN_network"):
        """
        Initialize the EFC CNN network based on the paper's architecture.
        
        Args:
            config: Configuration object with attributes like in_channels, num_classes, device, etc.
            name: Name of the network (default: "EFC_CNN_network").
        """
        super().__init__()

        # Define activation functions for compatibility with base Network
        self.activation_fn = activation_fn  # Used in modules
        self.out_activation_fn = out_activation_fn  # No activation before softmax (handled by loss)
        
        # Additional config attributes specific to CNN
        self.in_channels = config.in_channels  # e.g., 1 for grayscale, 3 for RGB
        self.num_classes = config.num_classes  # Number of output classes
        
        # Create the network architecture
        self.create_network()


    def create_network(self):
        """
        Build the CNN architecture: 4 conv modules + 1 FC layer, with separate BN layers.
        From: Vinyals et al. "Matching Networks for One Shot Learning" (2017)
        Args:
            config: Configuration object.
        """
        self.layers = nn.ModuleList()  # Layers for EFC modulation (conv and FC)
        self.bn_layers = nn.ModuleList()  # BatchNorm layers, not modulated
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Input channels for the first layer
        current_channels = self.in_channels
        
        # 4 Convolutional Modules
        for i in range(4):
            conv_layer = DFC_Conv_layer(
                in_channels=current_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=0,
                activation_fn=self.activation_fn()
            )
            bn_layer = nn.BatchNorm2d(64)
            self.layers.append(conv_layer)
            self.bn_layers.append(bn_layer)
            current_channels = 64
        
        # Fully Connected Layer
        self.layers.append(
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x):
        self.input = x
        self.bzs = x.shape[0]
        
        # Process through 4 convolutional modules
        for i in range(4):  # 4 modules
            conv_layer = self.layers[i]
            bn_layer = self.bn_layers[i]
            x = conv_layer(x)  # Convolution + ReLU
            x = bn_layer(x)    # Batch normalization
            x = self.pool(x)   # Max-pooling after each module
        
        # Flatten and apply final FC layer
        x = x.view(self.bzs, -1)  # [batch_size, 64]
        x = self.layers[-1](x)    # [batch_size, num_classes]
        
        self.y_hat = x
        return x


class JacobianInterface:
    def __init__(self, config):
        if config.mode == "ndi":
            self._inversion = self._non_dynamical_inversion
        else:
            self._inversion = self._dynamical_inversion
            self.dt = float(config.dt_di)
            self.apical_time_constant = self.dt
            self.time_constant_ratio = config.time_constant_ratio
            self.k_p = config.k_p
            self.tmax = config.tmax_di
            self.eps = float(config.eps)

            assert self.k_p > 0
            assert self.eps > 0

        if config.loss_fn == "mse":
            self._compute_error = self._compute_error_mse
            self._set_targets = self._set_targets_mse
        else:
            self._compute_error = self._compute_error_ce
            self._set_targets = self._set_targets_ce
            self._softmax = nn.Softmax(dim=1)

        self.tau = config.tau
        self.target_lr = float(config.target_lr)
        self.alpha = float(config.alpha_di)
        self.alpha_I = float(config.alpha_I)

        assert self.alpha > 0

    def backward(self, y):
        self._set_targets(y)
        self._inversion()

        for layer in self.layers:
            layer.backward()

    def _compute_error_mse(self, y_hat, y):
        return y - y_hat

    def _compute_error_ce(self, y_hat, y):
        return y - self._softmax(y_hat)

    def _set_targets_mse(self, y):
        """ MSE loss solution """
        self.targets = (1 - 2 * self.target_lr) * self.y_hat + 2 * self.target_lr * y
        self.output_size = self.targets.shape[1]

    def _set_targets_ce(self, y):
        """ CE loss solution """
        self.targets = self._softmax(self.y_hat) - self.target_lr * (self._softmax(self.y_hat) - y)
        self.output_size = self.targets.shape[1]

    def _calculate_layerwise_jacobians(self):
        """
        Compute the Jacobian J_{i,i-1} for each layer using DFC_layer's method.
        """
        Js = []
        for layer in self.layers:
            J = layer.compute_layerwise_jacobian()
            Js.append(J)
        return Js

    @torch.enable_grad()
    def _calculate_jeff_and_gammaeff(self):
        # Recompute forward with requires_grad and retain_grad on modulatable activations
        r_list = []
        r = self.input.detach().requires_grad_(True)  # Enable grad from the start
        for layer in self.layers:
            # For linear layers
            a = r @ layer.weights.t() + layer.bias.unsqueeze(0)
            r = layer.activation_fn(a)
            r.retain_grad()
            r_list.append(r)
        y = r_list[-1]
        out_dim = y.shape[1]  # Assume vector output (bzs, classes)

        J_eff = torch.zeros(self.bzs, out_dim, out_dim)
        gamma_eff = torch.zeros(self.bzs, out_dim)

        # Collect rows of cumulative Jacobians for each layer
        ji_rows_per_layer = [[] for _ in r_list]
        for k in range(out_dim):
            # Zero previous grads
            for r_i in r_list:
                if r_i.grad is not None:
                    r_i.grad.zero_()

            grad_outputs = torch.zeros_like(y)
            grad_outputs[:, k] = 1.0

            y.backward(gradient=grad_outputs, retain_graph=True)

            # Collect the k-th row for each layer
            for l, r_i in enumerate(r_list):
                grad_flat = r_i.grad.view(self.bzs, -1).clone()
                ji_rows_per_layer[l].append(grad_flat)

        # Now process per layer
        J_list = []
        gamma_list = []
        for l in range(len(r_list)):
            # Stack rows to form Ji_flat (bzs, out_dim, flat_dim)
            Ji_flat = torch.stack(ji_rows_per_layer[l], dim=1)
            J_list.append(Ji_flat)

            r_ff_flat = r_list[l].detach().view(self.bzs, -1)

            gamma_i = self._compute_fisher_modulation(self.layers[l], l)
            gamma_list.append(gamma_i)
            gamma_flat = gamma_i.view(self.bzs, -1) if not self._first_task else 0.0

            # Compute contribution to J_eff: Ji @ diag(r) @ Ji^T = (Ji_flat * r_ff_flat.unsqueeze(1)) @ Ji_flat.transpose(1, 2)
            temp = Ji_flat * r_ff_flat.unsqueeze(1)
            J_eff += torch.bmm(temp, Ji_flat.transpose(1, 2))

            # Compute contribution to gamma_eff: Ji @ (gamma ⊙ r)
            gamma_r_flat = (gamma_flat * r_ff_flat).unsqueeze(-1)
            gamma_eff += torch.bmm(Ji_flat, gamma_r_flat).squeeze(-1)

        return J_eff, gamma_eff, J_list, gamma_list

    def _calculate_full_jacobian(self):
        Js = [None] * len(self.layers)
        output_size = self.layer_sizes[-1]

        activations_derivatives = [
            layer.activation_derivative(layer.v_ff)
            for layer in self.layers
        ]

        # Last layer
        Js[-1] = activations_derivatives[-1].view(self.bzs, output_size, 1) * torch.eye(output_size)

        # Rest of the layers
        for i in range(len(self.layers) - 2, -1, -1):
            Js[i] = activations_derivatives[i].unsqueeze(1) * torch.matmul(
                Js[i+1], self.layers[i + 1].weights
            )

        return torch.cat(Js, dim=2), Js
    
    @torch.no_grad()
    def _calculate_psis(self, u):
        L = len(self.layers)
        psi_list = [None] * L

        # Derivatives per layer
        activations_derivatives = [layer.activation_derivative(layer.v_ff) for layer in self.layers]
        
        # Last layer
        psi = u * activations_derivatives[-1]
        psi_list[-1] = psi
        
        # Backward from second-to-last to first
        for i in range(L - 2, -1, -1):
            psi = (psi @ self.layers[i + 1].weights) * activations_derivatives[i]
            psi_list[i] = psi
        
        return psi_list


class FisherInterface:
    def __init__(self):
        self._fisher = {}  # Accumulated Fisher matrix
        self._theta_star = {}  # Latest parameter optima (theta_T^*)
        self._first_task = True
        
    def _calculate_fisher(self, dataloader):
        """Compute Fisher Information Matrix across entire dataset"""
        fisher = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)

        self.eval()

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # Log likelihood computation
            outputs = self(inputs)
            log_probs = F.log_softmax(outputs, dim=1)

            # Can also calculate log likelihood with targets possibly TODO double check what to use
            log_likelihood = (log_probs * targets).sum(dim=1)
            
            # Compute gradients
            self.zero_grad()
            log_likelihood.sum().backward()

            # Accumulate squared gradients
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n].data += p.grad.data ** 2

        # Normalize
        for n in fisher.keys():
            fisher[n] /= len(dataloader.dataset)
            
        return fisher

    def complete_task(self, dataloader):
        """Update Fisher and Bayesian posterior """
        current_fisher = self._calculate_fisher(dataloader)
        self._theta_star = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}

        if self._first_task:
            self._fisher = current_fisher
            self._first_task = False
        else:
            for n in self._fisher:
                self._fisher[n] += current_fisher[n]

    @torch.no_grad()
    def _compute_gamma(self, layer, i):
        if self._first_task:
            return torch.zeros((self.bzs, layer.weights.shape[0]))
        
        F_weights = self._fisher[f'layers.{i}._weights']
        F_bias = self._fisher[f'layers.{i}._bias']
        
        weight_diff = layer._weights - self._theta_star[f'layers.{i}._weights']
        bias_diff = layer._bias - self._theta_star[f'layers.{i}._bias']
        
        # Gamma: batched, activity-dependent for weights
        gamma = (layer.r_prev @ (F_weights * weight_diff).T) + (F_bias * bias_diff)
        
        # Fisher norm: per-output, no activity or batch sum
        fisher_norm = torch.sum(F_weights ** 2, dim=1) + (F_bias ** 2) + 1e-8
        fisher_norm = torch.sqrt(fisher_norm)
        
        return -self.beta * gamma / fisher_norm
    

    def _compute_fisher_gamma_conv(self, layer, i):
        """
        Compute Fisher-based modulation for convolutional layers.
        
        Args:
            layer: The convolutional layer (e.g., an EFC_Conv_layer instance).
            i (int): Layer index in the network.
        
        Returns:
            torch.Tensor: Modulation term gamma with shape [out_channels].
        """
        out_channels = layer.out_channels  # Number of output channels
        gamma = torch.zeros(out_channels)  # Shape: [out_channels]
        fisher_norm = torch.zeros(out_channels)  # Shape: [out_channels]

        for n, p in layer.named_parameters():
            full_name = f'layers.{i}.{n}'
            if p.requires_grad and full_name in self._fisher:
                base_gamma = self._fisher[full_name] * (p - self._theta_star[full_name])
                if 'weight' in n:
                    # Weights shape: [out_channels, in_channels, kernel_h, kernel_w]
                    # base_gamma shape: [out_channels, in_channels, kernel_h, kernel_w]
                    gamma += base_gamma.sum(dim=(1, 2, 3))  # Sum over in_channels and kernel dims
                    fisher_norm += torch.sum(self._fisher[full_name]**2, dim=(1, 2, 3))
                elif 'bias' in n:
                    # Bias shape: [out_channels]
                    # base_gamma shape: [out_channels]
                    gamma += base_gamma
                    fisher_norm += self._fisher[full_name]**2

        return - self.beta * gamma / (torch.sqrt(fisher_norm) + 1e-8)