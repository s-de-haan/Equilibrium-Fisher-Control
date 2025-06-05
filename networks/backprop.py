import torch
import torch.nn as nn
from typing import List, Optional

from networks.base import BaseNetwork, BaseTaskIncrementalNetwork

class BPNetwork(BaseNetwork):
    """
    Standard backpropagation neural network.
    Uses the unified base interface.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__(input_dim, hidden_dims, output_dim)

# Task-Incremental BP Network for Split-MNIST style tasks
class TaskILBPNetwork(BaseTaskIncrementalNetwork):
    """Task-incremental BP network."""
    pass