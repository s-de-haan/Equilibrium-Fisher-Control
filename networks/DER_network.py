import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from networks.network_interface import Network
from networks.layers import BP_layer
from networks.activation_function import Softplus, Linear


class ReplayBuffer:
    """
    Replay buffer that stores inputs, labels, AND logits (dark knowledge).

    Uses reservoir sampling to maintain a fixed-size buffer with uniform
    sampling probability across all seen samples.
    """

    def __init__(self, buffer_size, device="cpu"):
        self.buffer_size = buffer_size if buffer_size is not None else 500
        self.device = device

        self.buffer_x = None  # Input tensors
        self.buffer_y = None  # Labels (as class indices)
        self.buffer_logits = None  # Stored logits (dark knowledge)

        self.num_seen = 0  # Total samples seen (for reservoir sampling)
        self.current_size = 0  # Current number of samples in buffer

    def add(self, x, y, logits):
        """
        Add samples to buffer using reservoir sampling.

        Args:
            x: Input tensor [batch_size, ...]
            y: Labels [batch_size] (class indices, not one-hot)
            logits: Model outputs [batch_size, num_classes]
        """
        batch_size = x.size(0)

        # Initialize buffer on first add
        if self.buffer_x is None:
            x_shape = (self.buffer_size,) + x.shape[1:]
            logits_shape = (self.buffer_size,) + logits.shape[1:]

            self.buffer_x = torch.zeros(*x_shape, device=self.device)
            self.buffer_y = torch.zeros(
                self.buffer_size, dtype=torch.long, device=self.device
            )
            self.buffer_logits = torch.zeros(*logits_shape, device=self.device)

        for i in range(batch_size):
            self.num_seen += 1

            if self.current_size < self.buffer_size:
                # Buffer not full: add directly
                idx = self.current_size
                self.current_size += 1
            else:
                # Buffer full: reservoir sampling
                idx = np.random.randint(0, self.num_seen)
                if idx >= self.buffer_size:
                    continue  # Don't add this sample

            self.buffer_x[idx] = x[i].detach()
            self.buffer_y[idx] = (
                y[i].detach() if y.dim() == 1 else y[i].argmax().detach()
            )
            self.buffer_logits[idx] = logits[i].detach()

    def sample(self, batch_size):
        """
        Sample a batch from the buffer.

        Returns:
            x, y, logits tensors (or None if buffer is empty)
        """
        if self.current_size == 0:
            return None, None, None

        # Sample indices
        indices = np.random.choice(
            self.current_size, size=min(batch_size, self.current_size), replace=False
        )

        return (
            self.buffer_x[indices],
            self.buffer_y[indices],
            self.buffer_logits[indices],
        )

    def __len__(self):
        return self.current_size


class DER_network(Network):
    """
    Dark Experience Replay (DER) network.

    Based on Buzzega et al. (2020) "Dark Experience for General Continual
    Learning: a Strong, Simple Baseline" (NeurIPS 2020).

    Key idea: Store samples along with their logits (dark knowledge) in a
    replay buffer. During training, replay old samples and use MSE loss
    between current and stored logits for knowledge distillation.

    Loss:
        L = L_CE(current_batch) + α * L_MSE(replay_logits, stored_logits)

    This is "DER" from the paper. See DERpp_network for the variant that
    adds a cross-entropy term on replay samples.
    """

    def __init__(self, config, name="DER_network"):
        super().__init__(BP_layer, Softplus, Linear, config, name)

        # DER hyperparameters
        self.alpha = (
            getattr(config, "der_alpha", None) or 0.5
        )  # α: logit distillation weight
        self.buffer_size = (
            getattr(config, "buffer_size", None) or 500
        )  # Replay buffer size
        self.replay_batch_size = (
            getattr(config, "replay_batch_size", None)
            or getattr(config, "batch_size", None)
            or 64
        )

        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.buffer_size, device=self.device)

    def forward(self, x):
        """Standard forward pass, storing full output for replay."""
        self.input = x
        self.bzs = x.shape[0]

        full_output = x
        for layer in self.layers:
            full_output = layer(full_output)

        # Store full output for adding to buffer
        self._full_output = full_output

        # Apply task mask
        x = full_output[:, self.task_masks[self.task_id]]
        self.y_hat = x
        return x

    def backward(self, y):
        """
        Compute loss with DER replay and backpropagate.

        Also adds current batch to replay buffer.
        """
        # Current task loss
        loss = self.loss_fn(self.y_hat, y)

        # Replay loss (if buffer has samples)
        if self.buffer.current_size > 0:
            replay_loss = self._replay_loss()
            if replay_loss is not None:
                loss += replay_loss

        loss.backward()

        # Add current batch to buffer (with detached logits)
        # Convert y to class indices if one-hot
        y_indices = y.argmax(dim=1) if y.dim() > 1 and y.size(1) > 1 else y
        self.buffer.add(self.input, y_indices, self._full_output)

    def _replay_loss(self):
        """
        Compute DER replay loss: MSE between current and stored logits.
        """
        # Sample from buffer
        x_replay, y_replay, logits_stored = self.buffer.sample(self.replay_batch_size)

        if x_replay is None:
            return None

        # Forward pass on replay samples
        replay_output = x_replay
        for layer in self.layers:
            replay_output = layer(replay_output)

        # MSE loss between current logits and stored logits (dark knowledge)
        loss = self.alpha * F.mse_loss(replay_output, logits_stored)

        return loss

    def complete_task(self, dataloader=None):
        """Called after task completion (interface compatibility)."""
        pass

    def get_buffer_stats(self):
        """Return buffer statistics for monitoring."""
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "total_seen": self.buffer.num_seen,
        }


class DERpp_network(Network):
    """
    Dark Experience Replay++ (DER++) network.

    Extension of DER that adds a cross-entropy loss on replay samples.

    Loss:
        L = L_CE(current) + α * L_MSE(replay_logits, stored_logits)
                         + β * L_CE(replay_labels)

    The additional CE term helps maintain decision boundaries.
    """

    def __init__(self, config, name="DERpp_network"):
        super().__init__(BP_layer, Softplus, Linear, config, name)

        # DER++ hyperparameters
        self.alpha = (
            getattr(config, "der_alpha", None) or 0.5
        )  # α: logit distillation weight
        self.beta = getattr(config, "der_beta", None) or 0.5  # β: replay CE weight
        self.buffer_size = getattr(config, "buffer_size", None) or 500
        self.replay_batch_size = (
            getattr(config, "replay_batch_size", None)
            or getattr(config, "batch_size", None)
            or 64
        )

        # Initialize replay buffer
        self.buffer = ReplayBuffer(self.buffer_size, device=self.device)

    def forward(self, x):
        """Standard forward pass."""
        self.input = x
        self.bzs = x.shape[0]

        full_output = x
        for layer in self.layers:
            full_output = layer(full_output)

        self._full_output = full_output
        x = full_output[:, self.task_masks[self.task_id]]
        self.y_hat = x
        return x

    def backward(self, y):
        """Compute loss with DER++ replay and backpropagate."""
        # Current task loss
        loss = self.loss_fn(self.y_hat, y)

        # Replay losses
        if self.buffer.current_size > 0:
            replay_loss = self._replay_loss()
            if replay_loss is not None:
                loss += replay_loss

        loss.backward()

        # Add current batch to buffer
        y_indices = y.argmax(dim=1) if y.dim() > 1 and y.size(1) > 1 else y
        self.buffer.add(self.input, y_indices, self._full_output)

    def _replay_loss(self):
        """
        Compute DER++ replay loss: MSE on logits + CE on labels.
        """
        x_replay, y_replay, logits_stored = self.buffer.sample(self.replay_batch_size)

        if x_replay is None:
            return None

        # Forward pass on replay samples
        replay_output = x_replay
        for layer in self.layers:
            replay_output = layer(replay_output)

        # MSE loss on logits (dark knowledge distillation)
        loss_mse = self.alpha * F.mse_loss(replay_output, logits_stored)

        # CE loss on labels (maintain decision boundaries)
        loss_ce = self.beta * F.cross_entropy(replay_output, y_replay)

        return loss_mse + loss_ce

    def complete_task(self, dataloader=None):
        """Called after task completion."""
        pass

    def get_buffer_stats(self):
        """Return buffer statistics for monitoring."""
        return {
            "buffer_size": len(self.buffer),
            "buffer_capacity": self.buffer_size,
            "total_seen": self.buffer.num_seen,
        }
