from networks.BP_network import *
from networks.EWC_network import *
from networks.EFC_network import *
from networks.DFC_network import *
from networks.DynDFC_network import DynDFC_network  # ✅ NEW
from src.datasets import SplitMNIST
from src.trainers import TrainerCL
from src.utils import dotdict


def main():
    # Training configuration
    config = {
        "layers": [784, 400, 400, 2],
        "num_classes": 2,
        "lr": 1e-3,
        "eta_ff": 1e-3,        # ✅ NEW
        "eta_dyn": 0.1,        # ✅ NEW
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
        "k_i": 0.0,           # ✅ NEW
        "k_d": 0.0,           # ✅ NEW
        "eps": 1e-4,
        "save": False,
        "importance_ewc": 1.0,
        "beta_efc": 1000.0,
    }
    config = dotdict(config)

    # Load continual learning tasks
    tasks_dataloaders = SplitMNIST(config=config).get_all_tasks_dataloaders()

    # Instantiate model
    model = DynDFC_network(config=config)  # ✅ NEW

    # Launch continual learning trainer
    trainer = TrainerCL(model, tasks_dataloaders, config)
    trainer.train()


if __name__ == "__main__":
    main()
