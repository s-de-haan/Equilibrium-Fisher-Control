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
        self.m = torch.tensor(1)

    def set_m(self, m):
        self.m = m

    def reset_m(self):
        self.m = torch.tensor(1)

    def forward(self, x):
        return x

    def derivative(self, x):
        return torch.ones_like(x) * self.m


class mReLU(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.m = torch.tensor(1)

    def set_m(self, m):
        self.m = m

    def reset_m(self):
        self.m = torch.tensor(1)

    def forward(self, x):
        return x.clamp(min=0)

    def derivative(self, x):
        grad = torch.ones_like(x) * self.m
        grad[x < 0] = 0
        return grad
    
class Softplus(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.m = torch.tensor(1)
        self.beta = 5
        self.softplus = nn.Softplus(beta=self.beta)
        self.sigmoid = nn.Sigmoid()
    
    def set_m(self, m):
        self.m = m

    def reset_m(self):
        self.m = torch.tensor(1)

    def forward(self, x):
        return self.softplus(x)
    
    def derivative(self, x):
        return self.sigmoid(self.beta * x) * self.m
    
class SoftmaxLinear(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.m = torch.tensor(1)
        self.softmax = nn.Softmax(dim=1)
        
    def set_m(self, m):
        self.m = m
        
    def reset_m(self):
        self.m = torch.tensor(1)
        
    def forward(self, x):
        self.last_output = self.softmax(x)  # Store for derivative
        return x
        
    def derivative(self, x):
        sm = self.last_output
        # Creates diagonal matrix with softmax outputs
        diag = torch.diag_embed(sm)
        # Creates outer product of softmax outputs
        outer = sm.unsqueeze(2) * sm.unsqueeze(1)
        # Jacobian is (diag - outer) * m
        return (diag - outer) * self.m

class SoftmaxSoftplus(ActivationFunction):
    def __init__(self):
        super().__init__()
        self.m = torch.tensor(1)
        self.beta = 5
        self.softplus = nn.Softplus(beta=self.beta)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        
    def set_m(self, m):
        self.m = m
        
    def reset_m(self):
        self.m = torch.tensor(1)
        
    def forward(self, x):
        softplus_out = self.softplus(x)
        self.last_output = self.softmax(softplus_out)  # Store for derivative
        return softplus_out
        
    def derivative(self, x):
        sm = self.last_output
        # Softplus derivative
        sp_deriv = self.sigmoid(self.beta * x)
        # Softmax Jacobian
        diag = torch.diag_embed(sm)
        outer = sm.unsqueeze(2) * sm.unsqueeze(1)
        # Combine using chain rule
        return torch.bmm((diag - outer), sp_deriv.unsqueeze(2)) * self.m