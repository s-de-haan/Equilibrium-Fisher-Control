import argparse
import wandb
from omegaconf import OmegaConf
import multiprocessing

# Import unified implementations
from src.trainers import Trainer
from networks import (
    BPNetwork, EWCNetwork, DFCNetwork, EFCNetwork,
    TaskILBPNetwork, TaskILDFCNetwork,
    TaskILEWCNetwork, TaskILEFCNetwork
)

# Import existing dataloaders (they work as-is)
from src.dataloaders import TaskILMNIST, DomainILMNIST, ClassILMNIST
from src.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Unified continual learning training script")
    
    # Network architecture
    parser.add_argument("--layers", type=int, nargs='+', default=[784, 400, 400],
                        help="Network layer sizes")
    parser.add_argument("--lr", type=float, default=1.5e-6, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"])
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    
    # Environment settings
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--save", type=str2bool, default="false", help="Save model")
    
    # Method selection
    parser.add_argument("--method", type=str, default="efc", 
                        choices=["bp", "ewc", "dfc", "efc"],
                        help="Learning method")
    parser.add_argument("--setting", type=str, default="taskIL", 
                        choices=["taskIL", "domainIL", "classIL"],
                        help="Continual learning setting")
    parser.add_argument("--dataset", type=str, default="MNIST", 
                        choices=["MNIST", "CIFAR10"])
    
    # Method-specific parameters
    parser.add_argument("--mode", type=str, default="di", choices=["ndi", "di"])
    parser.add_argument("--importance_ewc", type=float, default=1.0, 
                        help="EWC importance")
    parser.add_argument("--beta_efc", type=float, default=5.0, 
                        help="EFC Fisher preservation strength")
    parser.add_argument("--dt_di", type=float, default=0.008, 
                        help="Dynamic inversion time step")
    parser.add_argument("--k_p", type=float, default=2.0, 
                        help="Proportional gain")
    parser.add_argument("--alpha_di", type=float, default=1e-4, 
                        help="Damping coefficient")
    parser.add_argument("--tmax_di", type=int, default=500, 
                        help="Max iterations for dynamic inversion")
    parser.add_argument("--eps", type=float, default=1e-4, 
                        help="Convergence threshold")
    
    # Experiment settings
    parser.add_argument("--run_name", type=str, default="unified_experiment")
    parser.add_argument("--debug", type=str2bool, default="false")
    parser.add_argument("--flatten_imgs", type=str, default="default")
    
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unknown arguments: {unknown}")
    return args


def get_model(config):
    """Create model based on method and setting."""
    input_dim = config.layers[0]
    hidden_dims = config.layers[1:-1]
    use_dynamic_inversion = True if config.mode == "di" else False
    
    if config.setting == "taskIL":
        # Task-incremental learning with 5 tasks, 2 classes each
        task_output_dims = [2] * 5
        
        if config.method == "bp":
            return TaskILBPNetwork(input_dim, hidden_dims, task_output_dims)
        elif config.method == "ewc":
            return TaskILEWCNetwork(input_dim, hidden_dims, task_output_dims, 
                                   importance=config.importance_ewc)
        elif config.method == "dfc":
            return TaskILDFCNetwork(input_dim, hidden_dims, task_output_dims,
                                   dt=config.dt_di, k_p=config.k_p, 
                                   alpha=config.alpha_di, max_iter=config.tmax_di,
                                   tol=config.eps)
        elif config.method == "efc":
            return TaskILEFCNetwork(input_dim, hidden_dims, task_output_dims,
                                   beta=config.beta_efc, dt=config.dt_di,
                                   k_p=config.k_p, alpha=config.alpha_di,
                                   tmax=config.tmax_di, eps=config.eps,
                                   use_dynamic_inversion=use_dynamic_inversion)
    else:
        # Regular continual learning (domainIL, classIL)
        # Determine output dimension based on dataset and setting
        if config.dataset == "MNIST":
            output_dim = 10  # MNIST has 10 classes (0-9)
        elif config.dataset == "CIFAR10":
            output_dim = 10  # CIFAR-10 has 10 classes
        else:
            # Fallback to config if specified
            output_dim = config.layers[-1] if len(config.layers) > 1 else 10
        
        if config.method == "bp":
            return BPNetwork(input_dim, hidden_dims, output_dim)
        elif config.method == "ewc":
            return EWCNetwork(input_dim, hidden_dims, output_dim,
                             importance=config.importance_ewc)
        elif config.method == "dfc":
            return DFCNetwork(input_dim, hidden_dims, output_dim,
                             dt=config.dt_di, k_p=config.k_p,
                             alpha=config.alpha_di, max_iter=config.tmax_di,
                             tol=config.eps)
        elif config.method == "efc":
            return EFCNetwork(input_dim, hidden_dims, output_dim,
                             beta=config.beta_efc, dt=config.dt_di,
                             k_p=config.k_p, alpha=config.alpha_di,
                             tmax=config.tmax_di, eps=config.eps,
                             use_dynamic_inversion=use_dynamic_inversion)
    
    raise ValueError(f"Unknown method: {config.method}")


def get_dataset(config):
    """Get dataset based on setting and dataset."""
    if config.setting == "taskIL" and config.dataset == "MNIST":
        return TaskILMNIST(config).get_all_tasks_dataloaders()
    elif config.setting == "domainIL" and config.dataset == "MNIST":
        return DomainILMNIST(config).get_all_tasks_dataloaders()
    elif config.setting == "classIL" and config.dataset == "MNIST":
        return ClassILMNIST(config).get_all_tasks_dataloaders()
    else:
        raise ValueError(f"Unsupported combination: {config.setting} + {config.dataset}")


def main():
    args = parse_args()
    
    # Handle wandb sweep parameters
    if wandb.run is not None:
        sweep_config = dict(wandb.config)
        for key, value in sweep_config.items():
            setattr(args, key, value)
    
    # Convert to OmegaConf
    config = OmegaConf.create(vars(args))
    
    # Handle multiprocessing
    if config.num_workers > 0:
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass
    
    print("Configuration:")
    print(OmegaConf.to_yaml(config))
    
    # Create model and dataset
    model = get_model(config)
    tasks_dataloaders = get_dataset(config)
    
    # Move model to device
    model = model.to(config.device)
    
    print(f"Created {config.method.upper()} model: {type(model).__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize wandb
    if not config.debug:
        if wandb.run is None:
            project_name = f"unified_cl_{config.setting}_{config.dataset}"
            wandb.init(
                project=project_name,
                name=config.run_name,
                entity="equilibrium-fisher-control",
                config=OmegaConf.to_container(config, resolve=True)
            )
    
    # Create unified trainer
    trainer = Trainer(
        model=model,
        tasks_dataloaders=tasks_dataloaders,
        config=config
    )
    
    try:
        # Train the model
        trainer.train()
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if not config.debug and wandb.run is not None:
            wandb.finish(exit_code=1)
        raise
    
    # Clean up
    if not config.debug and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()