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


class EFC_Conv_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation_fn=None, name="EFC_Conv_layer"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.activation_fn = activation_fn
        self.name = name
        
        # Note: in_features is not directly applicable for conv layers, but set for compatibility
        self.in_features = in_channels  # Simplification; could be adjusted based on input spatial size
        
        # Initialize convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            bias=True
        )
        # Assign weights and bias for base class compatibility
        self._weights = self.conv.weight  # Shape: [out_channels, in_channels, K_h, K_w]
        self._bias = self.conv.bias       # Shape: [out_channels]
        
        nn.init.kaiming_normal_(self._weights, mode='fan_in', nonlinearity='relu') # TODO check if this works approx with Softplus

    def forward(self, x):
        self.r_prev = x
        a = self.conv(x)
        self.linear_activations = a
        self.r = self.activation_fn(a)
        return self.r
    
    def backward(self):
        teaching_signal = self.r - self.r_ff  # Shape: [batch_size, out_channels, height_out, width_out]
        bsz = self.r_prev.size(0)

        # Compute weight gradient using batched correlation
        weight_grad = F.conv2d(
            self.r_prev.transpose(0, 1),      # [in_channels, batch_size, height_in, width_in]
            teaching_signal.transpose(0, 1),  # [out_channels, batch_size, height_out, width_out]
            padding=self.padding,
            stride=self.stride,
            groups=self.in_channels
        )

        self._weights.grad = - weight_grad / bsz
        self._bias.grad = - teaching_signal.sum(dim=(0, 2, 3)) / bsz 