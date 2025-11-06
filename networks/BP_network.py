from networks.network_interface import Network
from networks.layers import *
from networks.activation_function import *

class BP_network(Network):
    def __init__(self, config, name="BP_network") -> None:
        if "activation_fun" in config:
            if config["activation_fun"]=="Tanh":
                act_fun = Tanh
            else:
                act_fun = ReLU
        else:
            act_fun = ReLU
        super().__init__(BP_layer, act_fun, Linear, config, name)

    def backward(self, _):
        self.loss.backward()

    def complete_task(self, _):
        pass
