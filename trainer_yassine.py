import argparse
import wandb
from omegaconf import OmegaConf

from networks.BP_network import BP_network
from networks.DFC_network import DFC_network
from networks.EWC_network import EWC_network
from networks.EFC_network import EFC_network
from src.datasets import SplitMNIST
from src.dataloaders import TaskILMNIST, DomainILMNIST, ClassILMNIST
from src.trainers import WandBTrainerCL
from src.utils import str2bool
from train import parse_args

def get_model(model_name: str, config):
    """Get model based on name."""
    models = {
        "bp": BP_network,
        "dfc": DFC_network,
        "ewc": EWC_network,
        "efc": EFC_network
    }
    return models[model_name](config)

def main():
    args = parse_args()
    
    # Update args with sweep values if running under wandb:
    if wandb.run is not None:
        sweep_config = dict(wandb.config)
        for key, value in sweep_config.items():
            setattr(args, key, value)
    
    # Convert the Namespace to an OmegaConf config object.
    config = OmegaConf.create(vars(args))
    
    print("Final configuration:")
    print(OmegaConf.to_yaml(config))
    
    wandb.init(project="continual_learning_baselines", name=config.run_name,
               config=OmegaConf.to_container(config))
    
    model = get_model(config.method, config)
    # tasks_dataloaders = SplitMNIST(config).get_all_tasks_dataloaders()
    
    loader = 'class'
    if loader == 'class':
        tasks_dataloaders = ClassILMNIST(config).get_all_tasks_dataloaders()
        trainer = WandBTrainerCL(model, tasks_dataloaders, config)
    trainer.train()

if __name__ == "__main__":
    main()