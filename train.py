import argparse
import wandb
from omegaconf import OmegaConf
import os
from datetime import datetime
import multiprocessing

# Import our new network implementations
from networks.backprop import BP_Network, TaskIL_BP_Network
from networks.ewc import EWC_Network, TaskIL_EWC_Network
from networks.dfc import DFC_Network, TaskIL_DFC_Network
from networks.efc import EFC_Network, TaskIL_EFC_Network
from src.trainers import WandBTrainerCL, WandBTrainerCLTaskIL

# Import the dataloaders from existing code
from src.dataloaders import TaskILMNIST, DomainILMNIST, ClassILMNIST, TaskILCIFAR10, DomainILCIFAR10, ClassILCIFAR10
from src.utils import str2bool

def parse_args():
    parser = argparse.ArgumentParser(description="Train continual learning model using CLI args.")
    # Network architecture & training hyperparameters:
    parser.add_argument("--layers", type=int, nargs='+', default=[784, 400, 400, 2],
                        help="Network layer sizes (e.g., 784 400 400 2)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
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
    parser.add_argument("--clamp", type=str2bool, default="false", help="Whether to clamp")
    parser.add_argument("--beta_efc", type=float, default=5.0, help="Beta parameter for EFC")
    parser.add_argument("--target_lr", type=float, default=1e-2, help="Target learning rate for EFC")
    parser.add_argument("--alpha_di", type=float, default=1e-4, help="Alpha for dynamic inversion")
    parser.add_argument("--tau", type=float, default=0.008, help="tau parameter")
    
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
    parser.add_argument("--dataset", type=str, default="MNIST", choices=["MNIST", "CIFAR10"], help="Dataset to use")
    parser.add_argument("--flatten_imgs", type=str, default="default", choices=["default", "True", "False"], help="Whether to use stability gap")
    parser.add_argument("--setting", type=str, default="domainIL", choices=["domainIL", "taskIL", "classIL"], help="Setting to use")
    parser.add_argument("--run_name", type=str, default="default", help="Run name for wandb")
    parser.add_argument("--fisher_normalization", type=str2bool, default="false", help="Whether to normalize the Fisher matrix")
    parser.add_argument("--stability_gap", type=str2bool, default="false", help="Whether to compute stability gap")
    
    # You can add any other hyperparameters you need.
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown CLI arguments:", unknown)
    return args


def get_model(model_name: str, setting: str, config):
    """Get model based on name and setting."""
    # Extract dimensions from config
    input_dim = config.layers[0]  # First dimension (e.g., 784 for MNIST)
    hidden_dims = config.layers[1:-1]  # Middle dimensions (e.g., [400, 400])
    
    # Determine the output dimension based on setting
    if setting == "taskIL":
        output_dim = config.layers[-1]  # Last dimension (e.g., 2 for each task)
        task_output_dims = [output_dim] * 5  # Assuming 5 tasks
    else:
        # For domainIL and classIL, we need 10 outputs for MNIST / CIFAR10 (all classes)
        output_dim = 10

    if setting == "taskIL":
        # Task-incremental learning models
        if model_name == "bp":
            return TaskIL_BP_Network(
                input_dim,       # input_dim
                hidden_dims,     # hidden_dims
                task_output_dims # task_output_dims
            )
        elif model_name == "dfc":
            return TaskIL_DFC_Network(
                input_dim,       # input_dim
                hidden_dims,     # hidden_dims
                task_output_dims,# task_output_dims
                config
            )
        elif model_name == "ewc":
            return TaskIL_EWC_Network(
                input_dim,       # input_dim
                hidden_dims,     # hidden_dims
                task_output_dims,# task_output_dims
                config.importance_ewc
            )
        elif model_name == "efc":
            return TaskIL_EFC_Network(
                input_dim,       # input_dim
                hidden_dims,     # hidden_dims
                task_output_dims,# task_output_dims
                config
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    else:
        # Standard models for domain-incremental or class-incremental learning
        if model_name == "bp":
            return BP_Network(
                input_dim,      # input_dim
                hidden_dims,    # hidden_dims
                output_dim      # output_dim (e.g., 10 for full MNIST)
            )
        elif model_name == "dfc":
            return DFC_Network(
                input_dim,      # input_dim
                hidden_dims,    # hidden_dims
                output_dim,     # output_dim (e.g., 10 for full MNIST)
                config
            )
        elif model_name == "ewc":
            return EWC_Network(
                input_dim,      # input_dim
                hidden_dims,    # hidden_dims
                output_dim,     # output_dim (e.g., 10 for full MNIST)
                config.importance_ewc
            )
        elif model_name == "efc":
            return EFC_Network(
                input_dim,      # input_dim
                hidden_dims,    # hidden_dims
                output_dim,     # output_dim (e.g., 10 for full MNIST)
                config
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}")

def get_dataset(setting: str, dataset: str, config):
    """Get dataset based on setting."""
    if setting == "domainIL" and dataset == "MNIST":
        return DomainILMNIST(config).get_all_tasks_dataloaders()
    elif setting == "taskIL" and dataset == "MNIST":
        return TaskILMNIST(config).get_all_tasks_dataloaders()
    elif setting == "classIL" and dataset == "MNIST":
        return ClassILMNIST(config).get_all_tasks_dataloaders()
    elif setting == "domainIL" and dataset == "CIFAR10":
        return DomainILCIFAR10(config).get_all_tasks_dataloaders()
    elif setting == "taskIL" and dataset == "CIFAR10":
        return TaskILCIFAR10(config).get_all_tasks_dataloaders()
    elif setting == "classIL" and dataset == "CIFAR10":
        return ClassILCIFAR10(config).get_all_tasks_dataloaders()
    else:
        raise ValueError("Invalid setting or dataset")


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

    if config.num_workers > 0:
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    
    
    # Get model and datasets
    model = get_model(config.method, config.dataset, config)
    tasks_dataloaders = get_dataset(config.setting, config.dataset, config)
    
    # Set up WandB project
    project_name = f"{config.setting}_{config.dataset}_incremental_learning_baselines"
    
    # Initialize WandB if it's not already initialized
    if wandb.run is None:
        wandb.init(project=project_name, 
            name=config.run_name,
            entity="equilibrium-fisher-control",
            config=OmegaConf.to_container(config))
        
    # Train the model
    if config.setting == "taskIL":
        trainer = WandBTrainerCLTaskIL(model, tasks_dataloaders, config)
    else:
        trainer = WandBTrainerCL(model, tasks_dataloaders, config)
    trainer.train()

    # Save the model if requested
    if config.save:
        os.makedirs("models", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        trainer.save_model(f"models/{config.method}_{timestamp}_model.pt")

if __name__ == "__main__":
    main()