import torch
import torch.nn as nn
import numpy as np


class ModulationReLULayer(nn.Module):
    def __init__(self, size, a=1.2):
        super().__init__()
        self.size = size
        self.a = a
        self.control_signals = torch.ones(size)

    def forward(self, x):
        return self.control_signals * torch.relu(x)

    def set_control_signals(self, signals):
        self.control_signals = torch.clamp(signals, min=1.0 / self.a, max=self.a)

    def get_control_signals(self):
        return self.control_signals


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 8
        self.hidden_size = 20
        self.output_size = 2

        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.hidden_activations = ModulationReLULayer(self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        self.output_activations = ModulationReLULayer(self.output_size)

    def forward(self, x):
        x = self.flatten(x)
        h1 = self.layer1(x)
        h1_activated = self.hidden_activations(h1)
        out = self.layer2(h1_activated)
        out_activated = self.output_activations(out)
        return torch.softmax(out_activated, dim=-1)

    def set_control_signals(self, signals):
        hidden_signals, output_signals = (
            signals[:, : self.hidden_size],
            signals[:, self.hidden_size :],
        )
        self.hidden_activations.set_control_signals(hidden_signals)
        self.output_activations.set_control_signals(output_signals)

    def get_hidden_features(self, x):
        x = self.flatten(x)
        h1 = self.layer1(x)
        return self.hidden_activations(h1).detach().numpy()

    def get_weights(self):
        return [
            self.layer1.weight.detach().numpy(),
            self.layer2.weight.detach().numpy(),
        ]

    def reset_control_signals(self):
        self.hidden_activations.set_control_signals(torch.ones(self.hidden_size))
        self.output_activations.set_control_signals(torch.ones(self.output_size))


class ControlNet(nn.Module):
    def __init__(self, a=1.2):
        super(ControlNet, self).__init__()
        self.input_size = 30  # 8 input + 20 hidden + 2 output
        self.hidden_size = 40
        self.output_size = 22  # Control signals for hidden + output neurons
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.output_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.a = a

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x + 1)
        x = torch.clamp(x, min=1.0 / self.a, max=self.a)

        return x
