import torch
import torch.nn as nn
import torch.nn.functional as F

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
        return [layer.activations for layer in self.layers]

    @property
    def linear_activations(self):
        return [layer.linear_activations for layer in self.layers]

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
    
    def complete_task(self, dataloader):
        pass

class JacobianInterface:
    def __init__(self, config):
        if config.mode == "ndi":
            self._inversion = self._non_dynamical_inversion
        else:
            self._inversion = self._dynamical_inversion
            self.dt = config.dt_di
            self.time_constant_ratio = config.time_constant_ratio
            self.k_p = config.k_p
            self.tmax = config.tmax_di
            self.eps = config.eps

            assert self.k_p > 0
            assert self.eps > 0

        if config.loss_fn == "mse":
            self._compute_error = self._compute_error_mse
            self._set_targets = self._set_targets_mse
        else:
            self._compute_error = self._compute_error_ce
            self._set_targets = self._set_targets_ce
            self._softmax = nn.Softmax(dim=1)

        for i, layer in enumerate(self.layers):
            layer.tau = config.taus[i]

        self.target_lr = config.target_lr
        self.alpha = config.alpha_di

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

    def _calculate_full_jacobian(self):
        Js = [None] * len(self.layers)
        output_size = self.layer_sizes[-1]

        activations_derivatives = [
            layer.activation_derivative(layer.linear_activations)
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
            probs = torch.exp(log_probs)
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