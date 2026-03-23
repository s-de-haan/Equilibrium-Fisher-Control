from collections import OrderedDict
from typing import Any, Tuple
import argparse
import os

import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F

from networks.activation_function import ScaledSigmoid


class ModelOutput(OrderedDict):
    """Base ModelOutput class fixing the output type from the models. This class is inspired from
    the ``ModelOutput`` class from hugginface transformers library

    taken from clementchadebec github"""

    def __getitem__(self, k):
        if isinstance(k, str):
            self_dict = {k: v for (k, v) in self.items()}
            return self_dict[k]
        else:
            return self.to_tuple()[k]

    def __setattr__(self, name, value):
        super().__setitem__(name, value)
        super().__setattr__(name, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().__setattr__(key, value)

    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not ``None``.
        """
        return tuple(self[k] for k in self.keys())


class dotdict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_device():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return device


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

################################################
##       Useful functions for notebooks       ##
################################################

@torch.no_grad()
def evaluate(model, train_loader, test_loader, device):
    def run_epoch(data_loader, split_name):
        epoch_loss = 0
        total = 0
        correct = 0

        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)

            y_hat = model(X)
            loss = model.loss_fn(y_hat, y)

            epoch_loss += loss.item()
            total += y.size(0)
            correct += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()

            if torch.isnan(torch.tensor(epoch_loss)):
                raise ArithmeticError(f"NaN detected in {split_name} loss")

        epoch_loss /= len(data_loader)
        accuracy = 100 * correct / total
        return epoch_loss, accuracy

    train_loss, train_acc = run_epoch(train_loader, "train")
    test_loss, test_acc = run_epoch(test_loader, "test")

    print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.2f}%")
    print(f"Test  loss: {test_loss:.4f} | Test  acc: {test_acc:.2f}%")

    return (train_loss, train_acc), (test_loss, test_acc)


def plot_generated_inputs(generated_inputs, n_cols=8, title="Generated Samples", save=False, dir="plots/", filename="generated_samples.png"):
    """
    Plot generated input samples (e.g. from DFC generate_samples()).

    Args:
        generated_inputs (torch.Tensor): shape (batch_size, input_size)
        n_cols (int): number of columns in the grid
        title (str): plot title
    """
    # Move to CPU and detach from graph
    imgs = generated_inputs.detach().cpu()

    # Normalize to [0, 1] for display
    #imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min() + 1e-8)

    # Try to infer image dimensions (e.g. 28x28 for MNIST)
    img_size = int(imgs.shape[1] ** 0.5)
    imgs = imgs.view(-1, img_size, img_size)

    n_samples = imgs.shape[0]
    n_rows = (n_samples + n_cols - 1) // n_cols

    plt.figure(figsize=(n_cols * 1.5, n_rows * 1.5))
    for i in range(n_samples):
        plt.subplot(n_rows, n_cols, i + 1)
        #plt.imshow(imgs[i], cmap="gray", vmin=0.0, vmax=1.0)
        plt.imshow(imgs[i], cmap="gray", vmin=-0.4242, vmax=2.8215)
        plt.axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    if save:
        fname = dir + filename
        plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.show()


def compute_mnist_pixel_stats(data_root="./data"):
    """
    Loads MNIST (no normalization) and computes per-pixel mean and std.

    Returns:
        pixel_mean: tensor of shape (1, 28, 28)
        pixel_std: tensor of shape (1, 28, 28)
    """
    # Load raw MNIST (no normalization)
    train_dataset = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform= transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
    )

    # Stack all images into shape (60000, 1, 28, 28)
    all_images = torch.stack([img for img, _ in train_dataset], dim=0)

    # Compute per-pixel statistics
    pixel_mean = all_images.mean(dim=0)   # shape (1,28,28)
    pixel_std = all_images.std(dim=0)     # shape (1,28,28)

    return pixel_mean, pixel_std


def plot_receptive_fields(model, layer_idx=0, n_cols=8, n_rows=8, normalize=True):
    """
    Plot receptive fields (weights) of hidden units in a given layer.

    Args:
        model: trained DFC_network instance
        layer_idx: index of the layer to visualize (default: 0 for first hidden layer)
        n_cols: number of columns in the grid
        n_rows: number of rows in the grid
        normalize: whether to normalize weights to [0,1] for display
    """
    layer = model.layers[layer_idx]
    W = layer.weights.detach().cpu()  # shape (out_dim, in_dim)

    n_units = min(n_rows * n_cols, W.shape[0])
    img_size = int(W.shape[1] ** 0.5)  # assume square input, e.g. 28x28 for MNIST

    plt.figure(figsize=(1.5 * n_cols, 1.5 * n_rows))
    for i in range(n_units):
        rf = W[i].view(img_size, img_size)

        if normalize:
            rf_min, rf_max = rf.min(), rf.max()
            rf = (rf - rf_min) / (rf_max - rf_min + 1e-8)

        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(rf, cmap="gray")
        plt.axis("off")

    plt.suptitle(f"Receptive Fields - Layer {layer_idx}", fontsize=14)
    plt.tight_layout()
    plt.show()


@torch.no_grad()
def visualize_reconstructions(model, dataset, sm=True, device=None, n_samples=10, save=False, dir="plots/", filename="reconstructions.png"):
    """
    Visualize reconstructions from hidden and output layers using learned inverse weights.

    Args:
        model: trained DFC_network (with biases and activation_fn)
        V: decoder from hidden → input (shape: [input_dim, hidden_dim])
        V2: decoder from output → hidden (shape: [hidden_dim, output_dim])
        dataset: MNIST dataset or DataLoader
        device: torch.device
        n_samples: number of examples to visualize
    """
    device = device or next(model.parameters()).device
    model.eval()

    # === 1. Take samples ===
    if isinstance(dataset, torch.utils.data.DataLoader):
        x, _ = next(iter(dataset))
    else:
        x = torch.stack([dataset[i][0] for i in range(n_samples)])
    x = x[:n_samples].view(n_samples, -1).to(device)

    # === 2. Forward pass (with bias terms) ===
    layer1 = model.layers[0]
    layer2 = model.layers[1]

    # First layer: z1 = x @ W1^T + b1
    v1 = layer1(x)
    r1 = layer1.activation_fn(v1)

    # Second layer: z2 = r1 @ W2^T + b2
    v2 = layer2(r1)
    if sm:
        r2 = F.softmax(v2, dim=1)
    else:
        r2 = v2

    # === 3. Reconstructions ===
    layer1_fb = model.feedback_layers[0]
    layer2_fb = model.feedback_layers[1]

    # From hidden layer
    x_hat_from_hidden = torch.clamp(layer1_fb(r1), -0.4242 ,2.8215)

    # From output layer (through V2 → activation → V)
    r1_reconstructed = layer1.activation_fn(layer2_fb(r2))
    x_hat_from_output = torch.clamp(layer1_fb(r1_reconstructed), -0.4242 ,2.8215)

    # === 4. Plot results ===
    plt.figure(figsize=(10, 4))
    for i in range(n_samples):
        # Original
        plt.subplot(3, n_samples, i + 1)
        plt.imshow(x[i].view(28, 28).cpu(), cmap='gray', vmin=-0.4242, vmax=2.8215)
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original", fontsize=10)

        # From hidden layer
        plt.subplot(3, n_samples, n_samples + i + 1)
        plt.imshow(x_hat_from_hidden[i].view(28, 28).cpu(), cmap='gray', vmin=-0.4242, vmax=2.8215)
        plt.axis("off")
        if i == 0:
            plt.ylabel("From Hidden", fontsize=10)

        # From output layer
        plt.subplot(3, n_samples, 2 * n_samples + i + 1)
        plt.imshow(x_hat_from_output[i].view(28, 28).cpu(), cmap='gray', vmin=-0.4242, vmax=2.8215)
        plt.axis("off")
        if i == 0:
            plt.ylabel("From Output", fontsize=10)

    plt.suptitle("Reconstruction from Hidden and Output Layers (with biases)", fontsize=12)
    plt.tight_layout()
    if save:
        fname = dir + filename
        plt.savefig(fname, dpi=300, bbox_inches='tight') 
    plt.show()

    return x, x_hat_from_hidden, x_hat_from_output


@torch.no_grad()
def visualize_reconstructions_general(
    model,
    dataset,
    sm=False,
    device=None,
    n_samples=10,
    save=False,
    dir="plots/",
    filename="reconstructions.png",
):
    """
    Visualize reconstructions from every layer of the model.

    Assumptions:
        - model.layers is an ordered list of forward layers
        - model.feedback_layers decodes back toward the input
        - feedback_layers[0] maps first hidden -> input
        - feedback_layers[1] maps second hidden -> first hidden
        - ...
        - feedback_layers[-1] maps output -> previous layer

    Args:
        model: trained model
        dataset: dataset or DataLoader
        sm: if True, apply softmax to final layer activity before reconstruction
        device: torch device
        n_samples: number of examples to visualize
        save: whether to save figure
        dir: save directory
        filename: save file name
        clamp_range: tuple (min, max) or None
    Returns:
        x: original inputs
        activations: list of layer activations
        reconstructions: list of input reconstructions, one per layer
    """
    device = device or next(model.parameters()).device
    model.eval()
    scaled_sig = ScaledSigmoid()

    # === 1. Get samples ===
    if isinstance(dataset, torch.utils.data.DataLoader):
        x, _ = next(iter(dataset))
    else:
        x = torch.stack([dataset[i][0] for i in range(n_samples)])

    x = x[:n_samples].view(n_samples, -1).to(device)

    # === 2. Forward pass, store activations ===
    activations = []
    r = x
    n_layers = len(model.layers)

    for i, layer in enumerate(model.layers):
        v = layer(r)
        if i == n_layers - 1 and sm:
            r = F.softmax(v, dim=1)
        else:
            r = layer.activation_fn(v)
        activations.append(r)

    # === 3. Reconstruct from every layer back to input ===
    reconstructions = []

    for layer_idx in range(n_layers):
        r_recon = activations[layer_idx]

        # Decode down to input by chaining feedback maps:
        # layer_idx -> layer_idx-1 -> ... -> input
        for fb_idx in range(layer_idx, -1, -1):
            r_recon = model.feedback_layers[fb_idx](r_recon)
            if fb_idx > 0:
                r_recon += model.layers[fb_idx - 1].bias.unsqueeze(0)
                r_recon = model.layers[fb_idx - 1].activation_fn(r_recon)

        scaled_sig = ScaledSigmoid()
        r_recon = scaled_sig(r_recon)

        reconstructions.append(r_recon)

    # === 4. Infer image shape ===
    input_dim = x.shape[1]
    side = int(input_dim ** 0.5)
    is_square = side * side == input_dim

    # === 5. Plot ===
    n_rows = 1 + n_layers
    plt.figure(figsize=(1.5 * n_samples, 1.8 * n_rows))

    for i in range(n_samples):
        # Original
        plt.subplot(n_rows, n_samples, i + 1)
        if is_square:
            plt.imshow(x[i].view(side, side).cpu(), cmap="gray")
        else:
            plt.imshow(x[i].view(1, -1).cpu(), cmap="gray", aspect="auto")
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original")

        # Reconstructions from each layer
        for l in range(n_layers):
            plt.subplot(n_rows, n_samples, (l + 1) * n_samples + i + 1)
            if is_square:
                plt.imshow(reconstructions[l][i].view(side, side).cpu(), cmap="gray")
            else:
                plt.imshow(reconstructions[l][i].view(1, -1).cpu(), cmap="gray", aspect="auto")
            plt.axis("off")
            if i == 0:
                plt.ylabel(f"Layer {l+1}")

    plt.suptitle("Reconstructions from Every Layer")
    plt.tight_layout()

    if save:
        os.makedirs(dir, exist_ok=True)
        plt.savefig(os.path.join(dir, filename), dpi=300, bbox_inches="tight")

    plt.show()

    return x, activations, reconstructions