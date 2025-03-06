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
            conv_layer = EFC_Conv_layer(
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
        """
        Forward pass through the network.
        
        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, 28, 28].
        
        Returns:
            Tensor: Output tensor of shape [batch_size, num_classes].
        """
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

<<<<<<< HEAD
        # for i, layer in enumerate(self.layers):
        #     layer.tau = config.taus[i]

=======
        self.tau = config.tau
>>>>>>> ccb2bc18bfc387389d30ca671121a462bae70862
        self.target_lr = float(config.target_lr)
        self.alpha = float(config.alpha_di)

        assert self.alpha > 0

    def backward(self, y):
        self._set_targets(y)
        self._inversion()

        for layer in self.layers:
            layer.backward()
            layer.activation_fn.reset_modulation()

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
        # self.targets = self._softmax(self.y_hat) - self.target_lr * (self._softmax(self.y_hat) - y)
        self.targets = y
        self.output_size = self.targets.shape[1]

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

class FisherInterface:
    def __init__(self):
        self._means = {}
        self._fisher = {}  # Accumulated Fisher matrix
        self._means = {}  # Latest parameter optima (theta_T^*)
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
            # probs = torch.exp(log_probs)
            # log_likelihood = (log_probs * probs).sum(dim=1)

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
        """ Update accumulated Fisher and latest means after finishing a task. """
        current_fisher = self._calculate_fisher(dataloader)
        self._means = {n: p.data.clone() for n, p in self.named_parameters() if p.requires_grad}

        # Initialize for the first task else accumulate Fisher and update means
        if self._first_task:
            self._fisher = current_fisher
            self._first_task = False
        else:
            for n in self._fisher:
                self._fisher[n] += current_fisher[n]
        

    def _compute_fisher_modulation(self, layer, i):
        """Compute Fisher-based modulation for parameter preservation"""
        gamma = torch.zeros((layer.weights.shape[0]))
        fisher_norm = 0.0

        for n, p in layer.named_parameters():
            full_name = f'layers.{i}.{n}'
            if p.requires_grad:
                base_gamma = self._fisher[full_name] * (p - self._means[full_name])
                if 'weights' in n:
                    gamma += torch.sum(base_gamma, dim=1)
                    fisher_norm += torch.sum(self._fisher[full_name]**2, dim=1)                    
                elif 'bias' in n:
                    gamma += base_gamma
                    fisher_norm += self._fisher[full_name]**2
    
        return - self.beta * gamma / (torch.sqrt(fisher_norm) + 1e-8)

    
    def _compute_gamma(self, layer, i, normalize=False):
        """Compute Fisher-based modulation for parameter preservation"""
        gamma = torch.zeros((self.bzs, layer.weights.shape[0]))
        fisher_norm = 0.0

        for n, p in layer.named_parameters():
            full_name = f'layers.{i}.{n}'
            if p.requires_grad:
                base_gamma = self._fisher[full_name] * (p - self._means[full_name])
                if 'weights' in n:
                    gamma += (layer.r_prev @ base_gamma.T)
                    fisher_norm += torch.sum(self._fisher[full_name]**2, dim=1)
                elif 'bias' in n:
                    gamma += base_gamma
                    fisher_norm += self._fisher[full_name]**2
        
        if normalize:
            return - self.beta * gamma / (torch.sqrt(fisher_norm) + 1e-8)
        return - self.beta * gamma

    
    def _compute_fisher_modulation_conv(self, layer, i):
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
                base_gamma = self._fisher[full_name] * (p - self._means[full_name])
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