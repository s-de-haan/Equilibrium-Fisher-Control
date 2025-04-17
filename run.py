import torch
from torch import nn

from networks.BP_network import BP_network
from networks.EWC_network import EWC_network
from networks.DFC_network import DFC_network
from networks.EFC_network import EFC_network
from networks.DynDFC_network import DynDFC_network
from src.datasets import ConvMNIST
from src.trainers import Trainer
from src.utils import dotdict


def main():
    # Configuration dictionary
    cfg = dotdict({
        "layers": [784, 400, 400, 10],
        "num_classes": 10,
        "lr": 1e-3,
        "eta_ff": 1e-3,
        "eta_dyn": 0.1,
        "batch_size": 256,
        "epochs": 20,
        "mode": "di",
        "num_workers": 8,
        "loss_fn": "ce",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cuda:2",
        "output_dir": "./outputs",
        "seed": 1337,
        "taus": [0.02, 0.02, 0.02, 0.016, 0.01],
        "target_lr": 1e-2,
        "alpha_di": 3.0,
        "dt_di": 0.0016,
        "time_constant_ratio": 0.2,
        "tmax_di": 3000,
        "k_p": 1.0,
        "k_i": 0.0,
        "k_d": 0.0,
        "eps": 1e-3,
        "save": False,
    })

    # Prepare data loaders
    train_loader, test_loader = ConvMNIST(config=cfg).get_dataloaders()

    # Initialize DynDFC model
    model = DynDFC_network(config=cfg)

    # Training
    trainer = Trainer(model, train_loader, test_loader, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
