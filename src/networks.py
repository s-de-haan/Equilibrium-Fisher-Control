import torch
import torch.nn as nn


class ModulationReLULayer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.control_signals = torch.ones(size)

    def forward(self, x):
        return self.control_signals * torch.relu(x)

    def set_control_signals(self, signals):
        self.control_signals = signals

    def get_control_signals(self):
        return self.control_signals


class DFC_SSA_network(nn.Module):
    def __init__(self, config, name="DFC_SSA_network") -> None:
        super().__init__()
        self.create_network(config)

        self._target_lr = config.target_lr
        self._alpha_di = config.alpha_di
        self.name = name

    def create_network(self, config):
        layer_class = DFC_layer
        _layers = config.layers
        activation_fn = nn.ReLU()

        self.layers = nn.ModuleList()
        for i in range(len(_layers) - 2):
            self.layers.append(
                layer_class(
                    _layers[i],
                    _layers[i + 1],
                    activation_fn=activation_fn,
                )
            )
        self.layers.append(
            layer_class(
                _layers[-2],
                _layers[-1],
                activation_fn=Linear(),
            )
        )

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

    def backward(self, y):
        self._set_targets(y)
        self._non_dynamical_inversion()

        for layer in self.layers:
            layer.backward()

    def _set_targets(self, y):
        """MSE loss solution"""
        self.targets = (1 - 2 * self._target_lr) * self.y_hat + 2 * self._target_lr * y

    def _calculate_full_jacobian(self):
        Js = []

        activations_derivatives = [
            layer.activation_derivative(layer.linear_activations)
            for layer in self.layers
        ]
        bsz = self.layers[0].activations.shape[0]

        output_sz = self.layers[-1].out_features

        # Last layer
        Js.append(
            activations_derivatives[-1].view(bsz, output_sz, 1)
            * torch.eye(output_sz).repeat(bsz, 1, 1)
        )
        # Rest of the layers
        for i in range(len(self.layers) - 2, -1, -1):
            J = activations_derivatives[i].unsqueeze(1) * torch.matmul(
                Js[-1], self.layers[i + 1].weights
            )
            Js.append(J)

        Js.reverse()

        return torch.cat(Js, dim=2)

    def _non_dynamical_inversion(self):
        J = self._calculate_full_jacobian()
        J_T = J.transpose(1, 2)

        error = self.targets - self.y_hat
        error = error.unsqueeze(2)

        u = torch.linalg.solve(
            torch.matmul(J, J_T) + self._alpha_di * torch.eye(J.shape[1]), error
        )

        delta_v = torch.matmul(J_T, u).squeeze(-1)
        delta_vs = torch.tensor_split(
            delta_v,
            torch.cumsum(torch.tensor(self.layer_sizes[:-1]), dim=0).cpu(),
            dim=1,
        )

        rs = [self.input]

        for i, layer in enumerate(self.layers):
            v_ff = torch.matmul(rs[i], layer.weights.t())
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


class DFC_layer(nn.Module):
    def __init__(
        self, in_features, out_features, activation_fn=nn.ReLU(), name="DFC_layer"
    ) -> None:
        super(DFC_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.name = name

        self.activation_derivative = get_derivative(self.activation_fn)
        self.feedforward = nn.Sequential(
            nn.Linear(self.in_features, self.out_features), self.activation_fn
        )

        nn.init.kaiming_normal_(self.feedforward[0].weight)
        self._weights = self.feedforward[0].weight
        self._bias = self.feedforward[0].bias

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def shape(self):
        return self._weights.shape

    def forward(self, x):
        a = torch.matmul(x, self.weights.t())
        a += self.bias.unsqueeze(0).expand_as(a)
        self.activations = self.activation_fn(a)
        self.linear_activations = a

        return self.activations

    def backward(self):
        teaching_signal = self.r - self.r_ff

        bsz = self.r_prev.shape[0]
        weights_grad = -2 * 1.0 / bsz * teaching_signal.t().mm(self.r_prev)
        bias_grad = -2 * teaching_signal.mean(dim=0)

        self._weights.grad = weights_grad
        self._bias.grad = bias_grad


class BP_network(nn.Module):
    def __init__(self, config, name="BP_network") -> None:
        super().__init__()
        self.name = name

        _layers = config.layers
        activation_fn = nn.ReLU()

        self.layers = nn.ModuleList()
        for i in range(len(_layers) - 2):
            self.layers.append(
                BP_layer(
                    _layers[i],
                    _layers[i + 1],
                    activation_fn=activation_fn,
                )
            )
        self.layers.append(
            BP_layer(
                _layers[-2],
                _layers[-1],
                activation_fn=Linear(),
            )
        )

    def forward(self, x):
        self.input = x
        for layer in self.layers:
            x = layer(x)
        self.y_hat = x
        return x


class BP_layer(nn.Module):
    def __init__(
        self, in_features, out_features, activation_fn=nn.ReLU(), name="BP_layer"
    ) -> None:
        super(BP_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.name = name

        self.activation_derivative = get_derivative(self.activation_fn)
        self.feedforward = nn.Sequential(
            nn.Linear(self.in_features, self.out_features), self.activation_fn
        )

        nn.init.kaiming_normal_(self.feedforward[0].weight)
        self._weights = self.feedforward[0].weight
        self._bias = self.feedforward[0].bias

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def shape(self):
        return self._weights.shape

    def forward(self, x):
        a = torch.matmul(x, self.weights.t())
        a += self.bias.unsqueeze(0).expand_as(a)
        self.activations = self.activation_fn(a)
        self.linear_activations = a

        return self.activations


class Linear(nn.Module):
    def forward(self, x):
        return x


def derivative_sigmoid(x):
    return torch.mul(torch.sigmoid(x), 1.0 - torch.sigmoid(x))


def derivative_linear(x):
    return torch.ones_like(x)


def derivative_relu(x):
    grad = torch.ones_like(x)
    grad[x < 0] = 0
    return grad


def get_derivative(activation_fn):
    if isinstance(activation_fn, torch.nn.Sigmoid):
        return derivative_sigmoid
    elif isinstance(activation_fn, torch.nn.ReLU):
        return derivative_relu
    elif isinstance(activation_fn, Linear):
        return derivative_linear
    else:
        raise ValueError(f"Activation function {activation_fn} not supported")
