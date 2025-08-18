from networks.BP_network import *
from networks.EWC_network import *
from networks.EFC_network import *
from networks.DFC_network import *
from src.datasets import SplitMNIST
from src.trainers import TrainerCL
from src.utils import dotdict


def main():
    # Training configuration
    config = {
        "layers": [784, 400, 400, 2],
        "num_classes": 2,
        "lr": 1e-3,
        "batch_size": 256,
        "epochs": 4,
        "mode": "ndi",  # or "di"
        "num_workers": 8,
        "loss_fn": "ce", # "mse"
        "optimizer": "SGD",
        "scheduler": "CosineAnnealingLR",
        "device": "cpu",
        "output_dir": "./outputs",
        "seed": 0,
        "target_lr": 1.0, # needs to be < time_constant_ratio
        "alpha_di": 1e-3,
        "taus": [0.02, 0.016, 0.01],
        "dt_di": 0.0016,
        "psi_lr": 0.1,
        "alpha_psi": 0.0,
        "time_constant_ratio": 0.2, # this param can be merged with dt_di
        "tmax_di": 500,
        "k_p": 1.0,
        "eps": 1e-4, # there is an interplay between dt_di and eps and between target_lr and eps
        "save": False,
        "importance_ewc": 1.0, # ewc params
        "beta_efc": 0.0, # efc params
    }
    config = dotdict(config)

    # Load data
    tasks_dataloaders = SplitMNIST(config=config).get_all_tasks_dataloaders()

    # Train model
    model = EFC_network(config=config)

    trainer = TrainerCL(model, tasks_dataloaders, config)
    trainer.train()


if __name__ == "__main__":
    main()
