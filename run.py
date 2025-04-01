import torch.nn as nn

from networks.BP_network import *
from networks.EWC_network import *
from networks.DFC_network import *
from networks.EFC_network import *
from src.datasets import MNIST
from src.trainers import Trainer
from src.utils import dotdict
from src.serialize import clear_dumps
# TODO config manager from Xander's project

def main():
    # Training configuration
    config = {
        "layers": [784, 400, 400, 10],
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 20,
        "mode": "di",
        "num_workers": 4,
        "loss_fn": "ce",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cuda:0",
        "output_dir": "./outputs",
        "seed": 1337,
        # how much to scale the error. How big are the steps in modulation?
        # 1.0 means target = label, it's the scale of (label - output)
        "target_lr": 1.0,       # Updated to optimal value
        # should in [1e-3, 1e-2]
        "alpha_di": 3.0,        # 1/tau for tau=0.2
        "dt_di": 0.0016,         # Time step
        "time_constant_ratio": 0.2,  # tau
        "tmax_di": 500,
        # proportional term (don't touch)
        "k_p": 1.0,             # Optimal for G=1, tau=0.2
        "eps": 1e-4,            # convergence threshold
        "save": False,
    }
    config = dotdict(config)

    # Load data
    train_loader, test_loader = MNIST(config=config).get_dataloaders()

    # Train model
    clear_dumps()
    model = DFC_network(config=config)

    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()
