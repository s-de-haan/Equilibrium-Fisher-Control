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

def get_model(model_name: str, config):
    """Get model based on name."""
    models = {
        "bp": BP_network,
        "dfc": DFC_network,
        "ewc": EWC_network,
        "efc": EFC_network
    }
    return models[model_name](config)

def parse_args():
    parser = argparse.ArgumentParser(description="Train continual learning model using CLI args.")
    # Network architecture & training hyperparameters:
    parser.add_argument("--layers", type=int, nargs='+', default=[784, 400, 400, 2],
                        help="Network layer sizes (e.g., 784 400 400 2)")
    parser.add_argument("--lr", type=float, default=1.5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--mode", type=str, default="di", choices=["ndi", "di"],
                        help="whether to run with (di) or without (ndi) dynamic inversion")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--loss_fn", type=str, default='ce',
                        help="whether to train with cross entropy ('ce') or mean squared error ('mse') loss")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", help="Scheduler")
    
    # Environment settings hyperparameters
    parser.add_argument("--device", type=str, default="cuda", help="GPU/CPU device")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory for saving training and evaluation logs")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--save", type=str2bool, default="false", help="Whether to save the model (true/false)")
    
    # EFC-specific hyperparameters:
    parser.add_argument("--beta_efc", type=float, default=5.0, help="Beta parameter for EFC")
    parser.add_argument("--target_lr", type=float, default=1e-2, help="Target learning rate for EFC")
    parser.add_argument("--alpha_di", type=float, default=1e-4, help="Alpha for dynamic inversion")
    parser.add_argument("--taus", type=float, nargs='+', default=[0.01, 0.008, 0.006], help="tau parameters")
    
    # EWC-specific hyperparameters:
    parser.add_argument("--importance_ewc", type=float, default=1.0, help="Importance parameter for EWC")
    
    # Training method: ewc, efc, or bp.
    parser.add_argument("--method", type=str, default="efc", choices=["ewc", "efc", "bp", "dfc"],
                        help="Training method to use")
    
    # Additional parameters as needed:
    parser.add_argument("--dt_di", type=float, default=0.008, help="dt for dynamic inversion")
    parser.add_argument("--time_constant_ratio", type=float, default=0.2, help="Time constant ratio")
    parser.add_argument("--tmax_di", type=int, default=500, help="tmax for dynamic inversion")
    parser.add_argument("--k_p", type=float, default=2.0, help="Proportional gain for dynamic inversion")
    parser.add_argument("--eps", type=float, default=1e-4, help="Epsilon for convergence check")
    
    # You can add any other hyperparameters you need.
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown CLI arguments:", unknown)
    return args

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
    
    wandb.init(project="continual_learning_baselines", config=OmegaConf.to_container(config))
    
    model = get_model(config.method, config)
    # tasks_dataloaders = SplitMNIST(config).get_all_tasks_dataloaders()
    tasks_dataloaders = TaskILMNIST(config).get_all_tasks_dataloaders()
    trainer = WandBTrainerCL(model, tasks_dataloaders, config)
    trainer.train()

if __name__ == "__main__":
    main()