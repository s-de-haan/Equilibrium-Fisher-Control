import torch
import torch.nn as nn

from networks.layer_interface import Linear


class NetworkInterface(nn.Module):
    def __init__(self, layer_class, activation_fn, config, name):
        super().__init__()

        self.create_network(layer_class, activation_fn, config)
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
        for layer in self.layers:
            x = layer(x)
        self.y_hat = x
        return x

    def create_network(self, layer_class, activation_fn, config):
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
                activation_fn=Linear(),
            )
        )


class JacobianInterface(NetworkInterface):
    def __init__(self, layer_class, activation_fn, config, name) -> None:
        super().__init__(layer_class, activation_fn, config, name)

        if config.mode == "ndi":
            self._inversion = self._non_dynamical_inversion
        else:
            self._inversion = self._dynamical_inversion
            self.dt = config.dt_di
            self.apical_time_constant = self.dt
            self.time_constant_ratio = config.time_constant_ratio
            self.k_p = config.k_p
            self.tmax = config.tmax_di
            self.eps = config.eps

            assert self.k_p > 0
            assert self.apical_time_constant > 0
            assert self.eps > 0

        self.target_lr = config.target_lr
        self.alpha = config.alpha_di

    def backward(self, y):
        self._set_targets(y)
        self._inversion()

        for layer in self.layers:
            layer.backward()

    def _set_targets(self, y):
        """ MSE loss solution """
        self.targets = (1 - 2 * self.target_lr) * self.y_hat + 2 * self.target_lr * y
        self.bzs = self.targets.shape[0]
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
    
    def _broyden(self, Js_init=None): #TODO: implementation of Broyden's method for Jacobian updates.

        Js = [None] * len(self.layers)
        output_size = self.layer_sizes[-1]

        from scipy import optimize

        func = relu

        for i in range(len(self.layers) - 1, -1, -1):
            sol = optimize.broyden1(func, Js_init[i].cpu().numpy(), f_tol=1e-14, f_rtol=1e-14, verbose=True)
            Js[i] = torch.from_numpy(sol.copy())

        # sol = optimize.broyden1(func, Js_init, f_tol=1e-14, f_rtol=1e-14, verbose=True)
        return torch.cat(Js, dim=2), Js
    

def relu(x):
    #numpy based one TODO: horrible implementation, fix this.
    return torch.nn.ReLU()(torch.tensor(x)).cpu().numpy()