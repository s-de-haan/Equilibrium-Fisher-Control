import torch.nn as nn

from networks.network_interface import NetworkInterface
from networks.layers import BP_layer
from networks.activation_function import *

class BP_network(NetworkInterface):
    def __init__(self, config, name="BP_network") -> None:
        super().__init__(BP_layer, ReLU, Linear, config, name)