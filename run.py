import torch
import torch.nn as nn

from src.networks import DFC_SSA_network
from src.datasets import MNIST
from src.trainers import Trainer
from src.utils import dotdict, set_device

torch.set_grad_enabled(False)


def main():
    # Training configuration
    config = {
        "layers": [784, 256, 256, 10],
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 200,
        "runs": 10,
        "num_workers": 6,
        "loss_fn": nn.MSELoss(),
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cpu",
        "output_dir": "./outputs",
        "seed": 1337,
        "target_lr": 1e-4,
        "alpha_di": 1e-4,
    }
    config = dotdict(config)

    # Load data
    mnist = MNIST(config=config)
    train, eval = mnist.get_dataloaders(batch_size=config.batch_size)

    # Train model
    for _ in range(config.runs):
        model = DFC_SSA_network(config=config)

        trainer = Trainer(model, train, eval, config)
        trainer.train()


if __name__ == "__main__":
    main()
