import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

from networks.network_interface import NetworkInterface
from networks.layers import BP_layer
from networks.activation_function import ReLU, Linear

class EWC_network(NetworkInterface):
    def __init__(self, config, name="EWC_network"):
        super().__init__(BP_layer, ReLU, Linear, config, name)
        self.importance = config.importance_ewc
        self._means = {}
        self._precision_matrices = {}
        self._first_task = True
        
    def _calculate_batch_fisher(self):
        """Calculate Fisher Information for a single batch."""
        precision_matrices = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                precision_matrices[n] = torch.zeros_like(p)

        self.eval()
        output = self(self.input)
        label = output.argmax(1)
        loss = F.nll_loss(F.log_softmax(output, dim=1), label)
        loss.backward(retain_graph=True)
        
        for n, p in self.named_parameters():
            if p.requires_grad and p.grad is not None:
                precision_matrices[n].data += (p.grad.data ** 2) / self.bzs

        return precision_matrices

    def update_fisher(self):
        """Update Fisher Information Matrix using current batch."""
        batch_fisher = self._calculate_batch_fisher()  # Use only input data
        
        if not self._precision_matrices:  # Initialize if empty
            self._precision_matrices = batch_fisher
            return
        
        # Running average of Fisher Information
        for n in self._precision_matrices.keys():
            self._precision_matrices[n] = (
                0.95 * self._precision_matrices[n] + 
                0.05 * batch_fisher[n]
            )

    def store_task_parameters(self):
        """Store current parameters after task completion."""
        self._means = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                self._means[n] = p.data.clone()

    def ewc_loss(self):
        """Calculate EWC penalty term."""
        loss = 0
        if not self._first_task and self._means:
            for n, p in self.named_parameters():
                if p.requires_grad and n in self._means:
                    _loss = (
                        self._precision_matrices[n] * 
                        (p - self._means[n]) ** 2
                    ).sum()
                    loss += _loss
        return self.importance * loss

    def backward(self, y):
        # Update Fisher Information Matrix
        if not self._first_task:
            self.update_fisher()

        # Calculate gradients including EWC penalty
        loss = self.loss_fn(self.y_hat, y)
        if not self._first_task:
            loss += self.ewc_loss()
        
        loss.backward()

    def complete_task(self):
        """Call this at the end of each task."""
        if self._first_task:
            self._first_task = False
        self.store_task_parameters()