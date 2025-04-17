import torch
from networks.BP_network import BP_network
from networks.EWC_network import EWC_network
from networks.EFC_network import EFC_network
from networks.DFC_network import DFC_network
from networks.DynDFC_network import DynDFC_network
from src.datasets import SplitMNIST
from src.trainers import TrainerCL
from src.utils import dotdict


def main():
    # Configuration for continual learning
    cfg = dotdict({
        "layers": [784, 400, 400, 2],
        "num_classes": 2,
        "lr": 1e-3,
        "eta_ff": 1e-3,
        "eta_dyn": 0.1,
        "batch_size": 256,
        "epochs": 4,
        "mode": "di",
        "num_workers": 8,
        "loss_fn": "ce",
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cuda:5",
        "output_dir": "./outputs",
        "seed": 0,
        "target_lr": 1.0,
        "alpha_di": 1e-3,
        "taus": [0.02, 0.016, 0.01],
        "dt_di": 0.0016,
        "psi_lr": 0.1,
        "alpha_psi": 0.0,
        "time_constant_ratio": 0.2,
        "tmax_di": 500,
        "k_p": 1.0,
        "k_i": 0.0,
        "k_d": 0.0,
        "eps": 1e-4,
        "save": False,
        "importance_ewc": 1.0,
        "beta_efc": 1000.0,
    })

    # Load SplitMNIST tasks
    tasks = SplitMNIST(config=cfg).get_all_tasks_dataloaders()

    # Initialize DynDFC for continual learning
    model = DynDFC_network(config=cfg)

    # Train with continual learning framework
    trainer = TrainerCL(model, tasks, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
```
