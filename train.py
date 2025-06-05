# Updated training script with corrected EFC implementation

import argparse
import wandb
from omegaconf import OmegaConf
import os
from datetime import datetime
import multiprocessing
import torch

# Import the corrected EFC implementation
from networks.efc import EFC_Network_Wrapper, TaskIL_EFC_Network_Wrapper

# Import existing trainers - we'll need to modify the _train_step method
from src.trainers import WandBTrainerCL

# Import the dataloaders from existing code
from src.dataloaders import TaskILMNIST, DomainILMNIST, ClassILMNIST, TaskILCIFAR10, DomainILCIFAR10, ClassILCIFAR10
from src.utils import str2bool


class EFCTrainer(WandBTrainerCL):
    """Specialized trainer for EFC networks"""
    
    def _train_step(self, epoch: int) -> float:
        """Modified training step for EFC networks"""
        self.callback_handler.on_train_step_begin(
            training_config=self.config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        self.model.train()
        epoch_loss = 0
        
        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            # Use EFC forward training method
            if hasattr(self.model, 'forward_train'):
                y_hat = self.model.forward_train(X, y)
                loss = self.model.calculate_loss(y_hat, y)
            else:
                y_hat = self.model(X)
                loss = self.model.calculate_loss(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(training_config=self.config)

        epoch_loss /= len(self.train_loader)
        return epoch_loss


class EFCTaskILTrainer(EFCTrainer):
    """Specialized trainer for Task-Incremental EFC networks"""
    
    def _train_step(self, epoch: int, task_id: int) -> float:
        """Modified training step for Task-IL EFC networks"""
        self.callback_handler.on_train_step_begin(
            training_config=self.config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        # Set current task
        self.model.set_task(task_id)
        
        # Freeze previous task output heads
        self.model.freeze_previous_tasks()
        self.model.train()

        epoch_loss = 0
        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            # Use EFC forward training method with task ID
            y_hat = self.model.forward_train(X, y)
            loss = self.model.calculate_loss(y_hat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(training_config=self.config)

        epoch_loss /= len(self.train_loader)
        self.model.complete_task_and_freeze_output_head(task_id)
        return epoch_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train continual learning model using CLI args.")
    # Network architecture & training hyperparameters:
    parser.add_argument("--layers", type=int, nargs='+', default=[784, 400, 400, 2],
                        help="Network layer sizes (e.g., 784 400 400 2)")
    parser.add_argument("--lr", type=float, default=1.5e-6, help="Learning rate")
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
    parser.add_argument("--taus", type=float, nargs='+', default=[0.02, 0.016, 0.01], help="tau parameter")
    
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

    # Debug
    parser.add_argument("--debug", type=str2bool, default="false", help="Whether to debug experiment")
    
    # You can add any other hyperparameters you need.
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown CLI arguments:", unknown)
    return args


def get_model(config):
    """Get EFC model based on setting"""
    # Extract dimensions from config
    input_dim = config.layers[0]
    hidden_dims = config.layers[1:-1]
    
    if config.setting == "taskIL":
        return TaskIL_EFC_Network_Wrapper(
            config,
            num_tasks=5,
            task_output_size=2
        )
    else:
        # For domainIL and classIL, use 10 outputs for MNIST
        output_dim = 10
        return EFC_Network_Wrapper(
            input_dim,
            hidden_dims,
            output_dim,
            config
        )


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

    if config.num_workers > 0:
        try:
            multiprocessing.set_start_method('spawn')
        except RuntimeError:
            pass
    
    # Get model and datasets
    model = get_model(config)
    tasks_dataloaders = get_dataset(config.setting, config.dataset, config)
    
    # Move model to device
    model = model.to(config.device)
    
    # Set up WandB project
    project_name = f"{config.setting}_{config.dataset}_efc_corrected"
    
    # Initialize WandB if not debugging
    if not config.debug:
        if wandb.run is None:
            wandb.init(
                project=project_name, 
                name=config.run_name,
                entity="equilibrium-fisher-control",
                config=OmegaConf.to_container(config)
            )
    
    # Choose appropriate trainer
    if config.setting == "taskIL":
        trainer = EFCTaskILTrainer(model, tasks_dataloaders, config)
    else:
        trainer = EFCTrainer(model, tasks_dataloaders, config)
    
    try:
        trainer.train()
        print("Training completed successfully!")
        
        # Save the model if requested
        if config.save:
            os.makedirs("models", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            torch.save(model.efc_network.state_dict(), f"models/efc_corrected_{timestamp}.pt")
            
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        if not config.debug and wandb.run is not None:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    main()