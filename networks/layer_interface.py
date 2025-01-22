import torch
import torch.nn as nn


class LayerInterface(nn.Module):
    def __init__(self, in_features, out_features, activation_fn, name):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.name = name

        self.activation_derivative = self._get_derivative(self.activation_fn)

        self._create_init_layer()

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def shape(self):
        return self._weights.shape

    def _create_init_layer(self):
        self.feedforward = nn.Sequential(
            nn.Linear(self.in_features, self.out_features), self.activation_fn
        )

        nn.init.kaiming_normal_(self.feedforward[0].weight)
        self._weights = self.feedforward[0].weight
        self._bias = self.feedforward[0].bias

    def forward(self, x):
        a = torch.matmul(x, self.weights.t())
        a += self.bias.unsqueeze(0).expand_as(a)
        self.activations = self.activation_fn(a)
        self.linear_activations = a

        return self.activations

    def _derivative_sigmoid(self, x):
        return torch.mul(torch.sigmoid(x), 1.0 - torch.sigmoid(x))

    def _derivative_linear(self, x):
        return torch.ones_like(x)

    def _derivative_relu(self, x):
        grad = torch.ones_like(x)
        grad[x < 0] = 0
        return grad
    
    def _derivative_mrelu(self, x):
        grad = torch.zeros_like(x)
        grad[x > 0] = self.activation_fn.get_control_signals()[x > 0]
        return grad

    def _get_derivative(self, activation_fn):
        if isinstance(activation_fn, nn.Sigmoid):
            return self._derivative_sigmoid
        elif isinstance(activation_fn, nn.ReLU):
            return self._derivative_relu
        elif isinstance(activation_fn, Linear):
            return self._derivative_linear
        else:
            raise ValueError(f"Activation function {activation_fn} not supported")


class Linear(nn.Module):
    def forward(self, x):
        return x