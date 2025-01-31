import torch
import torch.nn as nn


class ActivationFunction(nn.Module):
    """Base class for activation functions with derivatives."""
    def forward(self, x):
        raise NotImplementedError

    def derivative(self, x):
        raise NotImplementedError


class Sigmoid(ActivationFunction):
    def forward(self, x):
        return torch.sigmoid(x)

    def derivative(self, x):
        sig = torch.sigmoid(x)
        return sig * (1 - sig)


class ReLU(ActivationFunction):
    def forward(self, x):
        return torch.relu(x)

    def derivative(self, x):
        grad = torch.ones_like(x)
        grad[x < 0] = 0
        return grad


class Linear(ActivationFunction):
    def forward(self, x):
        return x

    def derivative(self, x):
        return torch.ones_like(x)


class mLinear(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.m = 1

    def set_m(self, m):
        self.m = m

    def forward(self, x):
        return x * self.m

    def derivative(self, x):
        return torch.ones_like(x) * self.m


class mReLU(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.m = 1

    def set_m(self, m):
        self.m = m

    def forward(self, x):
        return x.clamp(min=0) * self.m

    def derivative(self, x):
        grad = torch.ones_like(x) * self.m
        grad[x < 0] = 0
        return grad