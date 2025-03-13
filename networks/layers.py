import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layer_interface import Layer

class DFC_layer(Layer):
    def __init__(self, in_features, out_features, activation_fn, name="DFC_layer"):
        super(DFC_layer, self).__init__(in_features, out_features, activation_fn, name)

    def backward(self):
        teaching_signal = self.r - self.r_ff

        self._weights.grad = -2 / self.r_prev.shape[0] * teaching_signal.t().mm(self.r_prev)
        self._bias.grad = -2 * teaching_signal.mean(dim=0)


class BP_layer(Layer):
    def __init__(
        self, in_features, out_features, activation_fn, name="BP_layer"
    ) -> None:
        super(BP_layer, self).__init__(in_features, out_features, activation_fn, name)


class DFC_Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, activation_fn=None, name="DFC_Conv_layer"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.activation_fn = activation_fn or nn.Identity()
        self.name = name
        
        # Convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True
        )
        
        # Alias weights and bias for compatibility with framework
        self._weights = self.conv.weight  # [out_channels, in_channels, kh, kw]
        self._bias = self.conv.bias       # [out_channels]
        
        # Initialize weights
        nn.init.kaiming_normal_(self._weights, mode='fan_in', nonlinearity='relu')
        
        # Attributes for dynamical system
        self.r_prev = None  # Input to the layer
        self.v_ff = None    # Pre-activation output
        self.r = None       # Post-activation output
        self.r_ff = None    # Feedforward target
        self.r_out = None   # Output after pooling (set by network)

    @property
    def weights(self):
        return self._weights

    @property
    def bias(self):
        return self._bias

    @property
    def activation_derivative(self):
        return self.activation_fn.derivative

    def forward(self, x):
        self.r_prev = x
        a = self.conv(x)
        self.v_ff = a
        self.r = self.activation_fn(a)
        return self.r

    def backward(self):
        """Compute gradients using teaching signal for weight updates."""
        teaching_signal = self.r - self.r_ff  # [batch_size, out_channels, h_out, w_out]
        bsz = self.r_prev.size(0)

        # Weight gradient via convolution
        weight_grad = F.conv2d(
            self.r_prev.transpose(0, 1),      # [in_channels, batch_size, h_in, w_in]
            teaching_signal.transpose(0, 1),  # [out_channels, batch_size, h_out, w_out]
            padding=self.padding,
            stride=self.stride,
            groups=self.in_channels
        )
        self._weights.grad = -weight_grad / bsz
        
        # Bias gradient
        self._bias.grad = -teaching_signal.sum(dim=(0, 2, 3)) / bsz