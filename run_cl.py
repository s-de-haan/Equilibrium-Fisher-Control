from networks.BP_network import *
from networks.EWC_network import *
from networks.EFC_network import *
from networks.DFC_network import *
from src.dataloaders import *
from src.trainers import TrainerCL
from src.utils import dotdict

def main():
    # Training configuration
    config = {
        "layers": [784, 400, 400, 10],
        "num_classes": 2,
        "lr": 1e-5,
        "batch_size": 256,
        "epochs": 20,
        "mode": "ndi",  # or "di"
        "num_workers": 8,
        "loss_fn": "ce", # "mse"
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "device": "cuda:5",
        "output_dir": "./outputs",
        "seed": 0,
        "target_lr": 1e-1, # needs to be < time_constant_ratio
        "alpha_di": 0.0017,
        "alpha_I": 0.0017,
        "tau": 0.032,
        "dt_di": 0.02,
        "psi_lr": 0.1,
        "alpha_psi": 0.0,
        "time_constant_ratio": 0.2, # this param can be merged with dt_di
        "tmax_di": 1000,
        "k_p": 2.0,
        "eps": 1e-4, # there is an interplay between dt_di and eps and between target_lr and eps
        "save": False,
        "importance_ewc": 0.0, # ewc params
        "beta_efc": 0, # efc params
        "flatten_imgs": True,
        "num_tasks": 5,
        "classes_per_task": 2,
        "setting": "taskIL", # domainIL, taskIL, classIL
    }
    config = dotdict(config)

    # Load data
    if config.setting == "domainIL":
        tasks_dataloaders = DomainILMNIST(config=config).get_all_tasks_dataloaders()
    elif config.setting == "taskIL":
        tasks_dataloaders = TaskILMNIST(config=config).get_all_tasks_dataloaders()
    elif config.setting == "classIL":
        tasks_dataloaders = ClassILMNIST(config=config).get_all_tasks_dataloaders()
    else:
        raise ValueError(f"Unknown setting: {config.setting}")

    # Train model
    model = EFC_network(config=config)

    trainer = TrainerCL(model, tasks_dataloaders, config)
    trainer.train()


if __name__ == "__main__":
    main()
