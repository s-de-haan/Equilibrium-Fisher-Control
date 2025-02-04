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
        self._fisher = {}
        self._first_task = True
    
    def _calculate_fisher(self, dataloader):
        """Compute Fisher Information Matrix across entire dataset"""
        fisher = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                fisher[n] = torch.zeros_like(p)

        self.eval()

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Log likelihood computation (either with probs or targets)
            outputs = self(inputs)
            log_probs = F.log_softmax(outputs, dim=1)
            probs = torch.exp(log_probs)  # Get actual probabilities
            log_likelihood = (log_probs * probs).sum(dim=1)

            # log_probs = F.log_softmax(outputs, dim=1)
            # log_likelihood = (log_probs * targets).sum(dim=1)

            # Compute gradients
            self.zero_grad()
            log_likelihood.sum().backward()

            # Accumulate squared gradients
            for n, p in self.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n].data += p.grad.data ** 2

        # Normalize
        for n in fisher.keys():
            fisher[n] /= len(dataloader.dataset)

        return fisher

    def ewc_loss(self, importance=1000.0):
        """Compute EWC regularization loss"""
        loss = 0.0
        
        for n, p in self.named_parameters():
            if n in self._means and n in self._fisher:
                loss += torch.sum(self._fisher[n] * (p - self._means[n]) ** 2)
        return importance * loss

    def backward(self, y):
        loss = self.loss_fn(self.y_hat, y)

        if not self._first_task:
            loss += self.ewc_loss()

        loss.backward()

    def complete_task(self, dataloader):
        """Store parameter means and compute Fisher Information Matrix"""
        if self._first_task:
            self._first_task = False
            
        # Store current parameter values
        self._means = {}
        for n, p in self.named_parameters():
            if p.requires_grad:
                self._means[n] = p.data.clone()
        
        # Compute Fisher Information Matrix
        self._fisher = self._calculate_fisher(dataloader)