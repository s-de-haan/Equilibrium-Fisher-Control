import torch
import torch.nn as nn

from networks.networks import *
from src.datasets import MNIST
from src.trainers import Trainer
from src.utils import dotdict, set_device


def main():
    # Training configuration
    config = {
        "layers": [784, 256, 256, 256, 10],
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 200,
        # "runs": 10,
        "mode": "di",  # or "di"
        "num_workers": 1,
        "loss_fn": nn.MSELoss(),
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cuda:0",
        "output_dir": "./outputs",
        "seed": 1337,
        "target_lr": 1e-2,
        "alpha_di": 1e-3,
        "dt_di": 0.02,  # dynamical inversion params
        "time_constant_ratio": 0.2,
        "tmax_di": 50,
        "k_p": 2.0,
        "save": False,
    }
    config = dotdict(config)

    # Load data
    train_loader, test_loader = MNIST(config=config).get_dataloaders()

    # Train model
    model = DFC_network(config=config)

    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
