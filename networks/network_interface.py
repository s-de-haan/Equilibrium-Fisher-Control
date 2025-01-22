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
