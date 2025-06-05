import datetime
import json
import torch
import os
import sys
import logging
from typing import Dict, Optional, Union

import wandb
from omegaconf import OmegaConf

from src.callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
from src.utils import dotdict

logger = logging.getLogger(__name__)


class Trainer:
    """
    Unified trainer that works with all network types (EFC, EWC, BP, DFC).
    Handles both regular continual learning and task-incremental learning.
    """
    
    def __init__(self, model, tasks_dataloaders, config, callbacks=None):
        self.model = model
        self.config = config
        self.tasks_dataloaders = tasks_dataloaders
        self.device = config.device
        self.save = config.save
        self.callbacks = callbacks
        
        # Determine if this is task-incremental learning
        self.is_task_incremental = hasattr(model, 'set_task') and hasattr(model, 'output_heads')
        
        # Initialize tracking variables
        self.task_accuracies = []
        self.global_step = 0
        self.current_task_step = 0  # Step within current task
        
        self._set_device(self.device)
        self.model.to(self.device)
        self._prepare_training()
        
        config_details = "\n".join([f" - {key}: {value}" for key, value in config.items()])
        logger.info(f"Training:\n{config_details}\n - model: {type(self.model).__name__}\n")
        logger.info(f"Task-incremental learning: {self.is_task_incremental}")

    def _set_device(self, device: str):
        self.device = torch.device(device)
        torch.set_default_device(self.device)

    def _prepare_training(self):
        self._set_seed(self.config.seed)
        self._set_optimizer()
        self._set_scheduler()
        self._set_output_dir()
        self._setup_logger()
        self._setup_callbacks()
        
        self.callback_handler.on_train_begin(training_config=self.config)

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _set_optimizer(self):
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.lr
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config.lr
            )
        else:
            raise NotImplementedError(f"Optimizer {self.config.optimizer} not implemented")

    def _set_scheduler(self):
        if hasattr(self.config, 'scheduler') and self.config.scheduler is not None:
            if self.config.scheduler == "CosineAnnealingLR":
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.config.epochs
                )
            # Add other schedulers as needed
        else:
            self.scheduler = None

    def _set_output_dir(self):
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self._training_signature = (
            str(datetime.datetime.now())[5:19].replace(" ", "_").replace(":", "-")
        )
        
        self.training_dir = os.path.join(
            self.config.output_dir,
            f"{type(self.model).__name__}_lr{self.config.lr}_{self._training_signature}",
        )
        
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir, exist_ok=True)

    def _setup_logger(self):
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        fh = logging.FileHandler(os.path.join(self.training_dir, "training.log"))
        fh.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)

    def _setup_callbacks(self):
        if self.callbacks is None:
            self.callbacks = [TrainingCallback()]
            
        self.callback_handler = CallbackHandler(
            callbacks=self.callbacks, model=self.model
        )
        
        self.callback_handler.add_callback(ProgressBarCallback())
        self.callback_handler.add_callback(MetricConsolePrinterCallback())

    def _log_metrics(self, metrics: Dict[str, float], step: int, task_id: Optional[int] = None):
        """Log metrics to WandB with proper step handling."""
        if wandb.run is not None:
            log_dict = {}
            if task_id is not None:
                # Add task-specific prefix
                for k, v in metrics.items():
                    log_dict[f"task_{task_id}/{k}"] = v
            else:
                log_dict = metrics.copy()
            
            wandb.log(log_dict, step=step)

    def _get_model_output(self, X, y, task_id=None):
        """Get model output with proper handling for different network types."""
        if hasattr(self.model, 'forward_train') and self.model.training:
            # EFC-style training
            if self.is_task_incremental and task_id is not None:
                if hasattr(self.model, 'efc_network'):
                    # Wrapper-style EFC
                    return self.model.forward_train(X, y)
                else:
                    # Direct EFC with task support
                    return self.model.forward_train(X, y, task_id=task_id)
            else:
                return self.model.forward_train(X, y)
        else:
            # Standard forward pass
            if self.is_task_incremental and task_id is not None:
                return self.model(X, task_id)
            else:
                return self.model(X)

    def _compute_loss(self, y_hat, y):
        """Compute loss with proper handling for different network types."""
        if hasattr(self.model, 'compute_loss'):
            return self.model.compute_loss(y_hat, y)
        elif hasattr(self.model, 'calculate_loss'):
            return self.model.calculate_loss(y_hat, y)
        else:
            # Fallback to CrossEntropy
            if y.dim() == 2:  # one-hot encoded
                y = y.argmax(dim=1)
            return torch.nn.functional.cross_entropy(y_hat, y)

    def _train_step(self, epoch: int, task_id: Optional[int] = None) -> float:
        """Single training step that works for both regular CL and Task-IL."""
        self.callback_handler.on_train_step_begin(
            training_config=self.config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        # Set up for task-incremental learning
        if self.is_task_incremental and task_id is not None:
            if hasattr(self.model, 'set_task'):
                self.model.set_task(task_id)
            if hasattr(self.model, 'freeze_previous_tasks'):
                self.model.freeze_previous_tasks()

        self.model.train()
        epoch_loss = 0
        total = 0
        correct = 0

        for batch_idx, (X, y) in enumerate(self.train_loader):
            X = X.to(self.device)
            y = y.to(self.device)

            # Forward pass
            y_hat = self._get_model_output(X, y, task_id)
            
            # Handle tuple return from EFC
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]
            
            # Compute loss
            loss = self._compute_loss(y_hat, y)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Track statistics
            epoch_loss += loss.item()
            
            # Calculate accuracy
            with torch.no_grad():
                if y_hat.dim() > 1:
                    predicted = y_hat.argmax(dim=1)
                else:
                    predicted = (y_hat > 0.5).long()
                    
                if y.dim() == 2:  # one-hot
                    targets = y.argmax(dim=1)
                else:
                    targets = y
                    
                total += y.size(0)
                correct += predicted.eq(targets).sum().item()

            self.callback_handler.on_train_step_end(training_config=self.config)

        # Finalize task if task-incremental
        if self.is_task_incremental and task_id is not None:
            if hasattr(self.model, 'complete_task_and_freeze_output_head'):
                self.model.complete_task_and_freeze_output_head(task_id)

        epoch_loss /= len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        # Log training metrics
        train_metrics = {
            "train/loss": epoch_loss,
            "train/accuracy": accuracy
        }
        self._log_metrics(train_metrics, self.global_step, task_id)
        
        return epoch_loss

    @torch.no_grad()
    def _test_step(self, epoch: int, task_id: Optional[int] = None) -> tuple:
        """Single test step that works for both regular CL and Task-IL."""
        self.callback_handler.on_test_step_begin(
            training_config=self.config,
            test_loader=self.test_loader,
            epoch=epoch,
        )

        self.model.eval()
        epoch_loss = 0
        total = 0
        correct = 0

        for X, y in self.test_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            # Forward pass (inference mode)
            if self.is_task_incremental and task_id is not None:
                y_hat = self.model(X, task_id)
            else:
                y_hat = self.model(X)
                
            # Handle tuple return
            if isinstance(y_hat, tuple):
                y_hat = y_hat[0]

            loss = self._compute_loss(y_hat, y)
            epoch_loss += loss.item()
            
            # Calculate accuracy
            if y_hat.dim() > 1:
                predicted = y_hat.argmax(dim=1)
            else:
                predicted = (y_hat > 0.5).long()
                
            if y.dim() == 2:  # one-hot
                targets = y.argmax(dim=1)
            else:
                targets = y
                
            total += y.size(0)
            correct += predicted.eq(targets).sum().item()

            self.callback_handler.on_test_step_end(training_config=self.config)

        epoch_loss /= len(self.test_loader)
        accuracy = 100.0 * correct / total

        # Log test metrics
        test_metrics = {
            "test/loss": epoch_loss,
            "test/accuracy": accuracy
        }
        self._log_metrics(test_metrics, self.global_step, task_id)

        return epoch_loss, accuracy

    def _test_seen_tasks(self, current_task_id: int):
        """Test on all seen tasks with proper step tracking."""
        task_accuracies = []
        
        for task_id in range(current_task_id + 1):
            logger.info(f"Testing on Task {task_id + 1}/{current_task_id + 1}")
            self.test_loader = self.tasks_dataloaders[task_id][1]

            epoch_test_loss, accuracy = self._test_step(0, task_id)
            task_accuracies.append(accuracy)

            logger.info(f"Task {task_id + 1} - Loss: {epoch_test_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Log aggregated metrics
        avg_accuracy = sum(task_accuracies) / len(task_accuracies)
        forgetting = max(task_accuracies) - min(task_accuracies) if len(task_accuracies) > 1 else 0.0
        
        aggregated_metrics = {
            "metrics/avg_accuracy": avg_accuracy,
            "metrics/forgetting": forgetting,
            "metrics/num_tasks_seen": current_task_id + 1
        }
        self._log_metrics(aggregated_metrics, self.global_step)
        
        self.task_accuracies.append(task_accuracies)

    def train(self):
        """Main training loop that handles both regular CL and Task-IL."""
        # Initialize WandB config
        if wandb.run is not None:
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            wandb.config.update(config_dict)

        metrics = dotdict()

        for task_id, (train_loader, test_loader) in enumerate(self.tasks_dataloaders):
            logger.info(f"Starting Task {task_id + 1}/{len(self.tasks_dataloaders)}")

            self.train_loader = train_loader
            self.test_loader = test_loader
            self.current_task_step = 0
            
            self.callback_handler.on_task_begin(
                training_config=self.config, task_id=task_id + 1
            )

            # Training epochs for current task
            for epoch in range(1, self.config.epochs + 1):
                self.callback_handler.on_epoch_begin(
                    training_config=self.config,
                    epoch=epoch,
                    train_loader=self.train_loader,
                    test_loader=self.test_loader
                )

                # Training step
                if self.is_task_incremental:
                    epoch_train_loss = self._train_step(epoch, task_id)
                else:
                    epoch_train_loss = self._train_step(epoch)
                
                metrics.epoch_train_loss = epoch_train_loss

                # Test step
                if self.test_loader is not None:
                    if self.is_task_incremental:
                        epoch_test_loss, accuracy = self._test_step(epoch, task_id)
                    else:
                        epoch_test_loss, accuracy = self._test_step(epoch)
                    
                    metrics.epoch_test_loss = epoch_test_loss
                    metrics.accuracy = accuracy

                self.callback_handler.on_epoch_end(training_config=self.config)
                self.callback_handler.on_log(
                    self.config,
                    metrics,
                    logger=logger,
                    epoch=epoch,
                )
                
                self.global_step += 1
                self.current_task_step += 1

            # Test on all seen tasks after completing current task
            self._test_seen_tasks(task_id)
            
            self.callback_handler.on_task_end(
                training_config=self.config, task_id=task_id + 1
            )

            # Complete task (for continual learning methods that need it)
            if hasattr(self.model, 'complete_task'):
                if hasattr(self.model, 'efc_network'):
                    # Wrapper-style networks
                    self.model.complete_task(train_loader, self.device)
                else:
                    # Direct networks
                    self.model.complete_task(train_loader)
            
            # Reset optimizer for next task
            self._set_optimizer()

        # Final logging
        if wandb.run is not None and len(self.task_accuracies) > 0:
            final_avg_accuracy = sum(self.task_accuracies[-1]) / len(self.task_accuracies[-1])
            final_forgetting = max(self.task_accuracies[0]) - min(self.task_accuracies[-1]) if len(self.task_accuracies) > 0 else 0.0
            
            wandb.run.summary.update({
                "final_avg_accuracy": final_avg_accuracy,
                "final_forgetting": final_forgetting
            })

        if self.save:
            self._save_model()

    def _save_model(self):
        """Save model and config."""
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)

        torch.save(self.model.state_dict(), os.path.join(self.training_dir, "model.pt"))

        with open(os.path.join(self.training_dir, "config.json"), "w") as fp:
            json.dump(OmegaConf.to_container(self.config, resolve=True), fp, indent=2)

        self.callback_handler.on_save(self.config)