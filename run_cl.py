import torch
import torch.nn as nn

from networks.networks import *
from src.datasets import SplitMNIST
from src.trainers import TrainerCL
from src.utils import dotdict


def main():
    # Training configuration
    config = {
        "layers": [784, 256, 256, 256, 2],
        "lr": 1e-3,
        "batch_size": 128,
        "epochs": 100,
        # "runs": 10,
        "mode": "ndi",  # or "di"
        "num_workers": 1,
        "loss_fn": nn.MSELoss(),
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cpu",
        "output_dir": "./outputs",
        "seed": 1337,
        "target_lr": 1e-2,
        "alpha_di": 1e-3,
        "dt_di": 0.02,  # dynamical inversion params
        "time_constant_ratio": 0.2,
        "tmax_di": 500,
        "k_p": 2.0,
        "eps": 1e-4,
        "save": False,
    }
    config = dotdict(config)

    # Load data
    tasks_dataloaders = SplitMNIST(config=config).get_all_tasks_dataloaders()

    # Train model
    model = DFC_network(config=config)

    trainer = TrainerCL(model, tasks_dataloaders, config)
    trainer.train()


if __name__ == "__main__":
    main()
