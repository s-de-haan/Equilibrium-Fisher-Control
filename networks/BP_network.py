from networks.network_interface import Network
from networks.layers import BP_layer
from networks.activation_function import *

class BP_network(Network):
    def __init__(self, config, name="BP_network") -> None:
        super().__init__(BP_layer, ReLU, Linear, config, name)

    def backward(self, _):
        self.loss.backward()