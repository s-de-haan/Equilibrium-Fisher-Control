import torch.nn as nn

from networks.BP_network import *
from networks.EWC_network import *
from networks.DFC_network import *
from networks.EFC_network import *
from networks.DynDFC_network import DynDFC_network  # ✅ NEW
from src.datasets import *
from src.trainers import Trainer
from src.utils import dotdict


def main():
    # Training configuration
    config = {
        "layers": [784, 400, 400, 10],
        "num_classes": 10,
        "lr": 1e-3,
        "eta_ff": 1e-3,             # ✅ NEW
        "eta_dyn": 0.1,             # ✅ NEW
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
        "k_i": 0.0,                 # ✅ NEW
        "k_d": 0.0,                 # ✅ NEW
        "eps": 1e-3,
        "save": False,
    }
    config = dotdict(config)

    # Load data
    train_loader, test_loader = ConvMNIST(config=config).get_dataloaders()

    # Instantiate DynDFC model
    model = DynDFC_network(config=config)  # ✅ Now using your updated model

    # Launch training
    trainer = Trainer(model, train_loader, test_loader, config)
    trainer.train()


if __name__ == "__main__":
    main()

