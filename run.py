import torch
import torch.nn as nn

from networks.networks import DFC_SSA_network, DFC_SSA_Mult_network, BP_network
from src.datasets import MNIST
from src.trainers import Trainer
from src.utils import dotdict, set_device

def main():
    # Training configuration
    config = {
        "layers": [784, 256, 256, 256, 10],
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 200,
        # "runs": 10,
        "num_workers": 1,
        "loss_fn": nn.MSELoss(),
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cpu",
        "output_dir": "./outputs",
        "seed": 1337,
        "target_lr": 1e-2,
        "alpha_di": 1e-3,
    }
    config = dotdict(config)

    # Load data
    train_loader, test_loader = MNIST(config=config).get_dataloaders()

    # Train model
    model = DFC_SSA_Mult_network(config=config)

    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
