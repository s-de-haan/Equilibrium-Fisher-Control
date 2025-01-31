import torch
import torch.nn as nn
import random

from networks.layer_interface import LayerInterface


class DFC_layer(LayerInterface):
    def __init__(self, in_features, out_features, activation_fn, name="DFC_layer"):
        super(DFC_layer, self).__init__(in_features, out_features, activation_fn, name)

    def backward(self):
        teaching_signal = self.r - self.r_ff
        
        bsz = self.r_prev.shape[0]
        weights_grad = -2 * 1.0 / bsz * teaching_signal.t().mm(self.r_prev)
        bias_grad = -2 * teaching_signal.mean(dim=0)

        self._weights.grad = weights_grad
        self._bias.grad = bias_grad

class BP_layer(LayerInterface):
    def __init__(
        self, in_features, out_features, activation_fn, name="BP_layer"
    ) -> None:
        super(BP_layer, self).__init__(in_features, out_features, activation_fn, name)
