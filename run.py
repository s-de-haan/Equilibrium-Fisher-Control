import torch.nn as nn

from networks.BP_network import *
from networks.EWC_network import *
from networks.DFC_network import *
from networks.EFC_network import *
from src.dataloaders import *
from src.trainers import Trainer
from src.utils import dotdict

def main():
    # Training configuration
    config = {
        "layers": [784, 400, 400, 10],
        "num_classes": 10,
        "lr": 1e-4,
        "batch_size": 256,
        "epochs": 20,
        "mode": "ndi",
        "num_workers": 8,
        "loss_fn": "ce",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cpu",
        "output_dir": "./outputs",
        "seed": 1337,
        "taus": [0.02, 0.02, 0.02, 0.016, 0.01],
        "target_lr": 1e-2,       # Updated to optimal value
        "alpha_di": 0.0017,        # 1/tau for tau=0.2
        "alpha_I": 0.0017,
        "dt_di": 0.02,         # Time step
        "time_constant_ratio": 0.2,  # tau
        "tmax_di": 500,
        "k_p": 2.0,             # Optimal for G=1, tau=0.2
        "eps": 1e-4,
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
