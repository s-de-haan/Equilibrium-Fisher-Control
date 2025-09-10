import datetime
import json
import torch
import os
import sys
import logging

import wandb
from omegaconf import OmegaConf
from typing import Dict, Optional

from networks.network_interface import FisherInterface
from src.callbacks import (
    CallbackHandler,
    MetricConsolePrinterCallback,
    ProgressBarCallback,
    TrainingCallback,
)
from src.utils import dotdict

logger = logging.getLogger(__name__)


class TrainerInterface:
    def __init__(self, model, config, callbacks=None):
        self.model = model
        self.config = config
        self.device = config.device
        self.save = config.save
        self.callbacks = callbacks
        self.setting = config.setting

        self._set_device(self.device)
        self.model.to(self.device)

        self._prepare_training()

        self.callback_handler.on_train_begin(training_config=self.config)

        config_details = "\n".join([f" - {key}: {value}" for key, value in config.items()])
        logger.info(msg=f"Training:\n{config_details}\n - model: {self.model.name}\n")

    def _set_device(self, device: str):
        self.device = torch.device(device)
        torch.set_default_device(self.device)

    def _save_model(self):
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)

        torch.save(self.model.state_dict(), os.path.join(self.training_dir, "model.pt"))

        with open(os.path.join(self.training_dir, "config.json"), "w") as fp:
            json.dump(self.config, fp)

        self.callback_handler.on_save(self.config)

    def _setup_logger(self):
        # Create a logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Create file handler which logs even debug messages
        fh = logging.FileHandler(os.path.join(self.training_dir, "training.log"))
        fh.setLevel(logging.INFO)

        # Create console handler with a higher log level
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter("%(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

    def _prepare_training(self):
        self._set_seed(self.config.seed)
        self._set_optimizer()
        self._set_scheduler()
        self._set_output_dir()
        self._setup_logger()
        self._setup_callbacks()

    def _set_seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _set_optimizer(self):
        if self.config.optimizer == "Adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config["lr"],
                betas=(0.9, 0.999),
                eps=5.83238643406511e-07
            )
        elif self.config.optimizer == "SGD":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.config["lr"]
            )
        else:
            raise NotImplementedError

    def _set_scheduler(self):
        if self.config.scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=self.config.lr, gamma=self.config.gamma
            )
        elif self.config.scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.gamma,
                patience=self.config.patience,
                verbose=True,
            )
        elif self.config.scheduler == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler is None:
            pass
        else:
            raise NotImplementedError

    def _set_output_dir(self):
        self.output_dir = self.config["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)

        self._training_signature = (
            str(datetime.datetime.now())[5:19].replace(" ", "_").replace(":", "-")
        )

        training_dir = os.path.join(
            self.config.output_dir,
            f"{self.model.name}_lr{self.config.lr}_{self._training_signature}",
        )

        self.training_dir = training_dir

        if not os.path.exists(training_dir):
            os.makedirs(training_dir, exist_ok=True)

    def _setup_callbacks(self):
        if self.callbacks is None:
            self.callbacks = [TrainingCallback()]

        self.callback_handler = CallbackHandler(
            callbacks=self.callbacks, model=self.model
        )

        self.callback_handler.add_callback(ProgressBarCallback())
        self.callback_handler.add_callback(MetricConsolePrinterCallback())

    def test_step(self, epoch):
        raise NotImplementedError("test_step must be implemented in a subclass.")

    def train(self):
        raise NotImplementedError("train must be implemented in a subclass.")

    def _train_step(self, epoch: int):
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

            y_hat = self.model(X)
            loss = self.model.calculate_loss(y_hat, y)

            self.optimizer.zero_grad()
            self.model.backward(y)
            self.optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")
            
            self.callback_handler.on_train_step_end(training_config=self.config)

        epoch_loss /= len(self.train_loader)

        return epoch_loss
    
    
    @torch.no_grad()
    def _test_step(self, epoch, task_id):
        self.callback_handler.on_test_step_begin(
            training_config=self.config,
            test_loader=self.test_loader,
            epoch=epoch,
        )

        epoch_loss = 0
        total = 0
        correct = 0

        if self.setting == "taskIL":
            self.model.task_id = task_id

        for X, y in self.test_loader:
            X = X.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(X)

            loss = self.model.loss_fn(y_hat, y)

            epoch_loss += loss.item()
            total += y.size(0)
            correct += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in test loss")
            
            self.callback_handler.on_test_step_end(training_config=self.config)

        epoch_loss /= len(self.test_loader)
        accuracy = 100 * correct / total

        return epoch_loss, accuracy
    

class Trainer(TrainerInterface):
    def __init__(self, model, train_loader, test_loader, config, callbacks=None):
        super().__init__(model, config, callbacks)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self):
        self.callback_handler.on_train_begin(training_config=self.config)
        metrics = dotdict()

        for epoch in range(1, self.config.epochs + 1):
            self.callback_handler.on_epoch_begin(
                training_config=self.config,
                epoch=epoch,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
            )

            epoch_train_loss = self._train_step(epoch)
            metrics.epoch_train_loss = epoch_train_loss

            if self.test_loader is not None:
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

        if self.save:
            self._save_model()


class TrainerCL(TrainerInterface):
    def __init__(self, model, tasks_dataloaders, config, callbacks=None):
        super().__init__(model, config, callbacks)
        self.tasks_dataloaders = tasks_dataloaders

    def train(self):
        self.callback_handler.on_train_begin(training_config=self.config)
        metrics = dotdict()

        for task_id, (train_loader, test_loader) in enumerate(self.tasks_dataloaders):
            logger.info(f"Starting Task {task_id + 1}/{len(self.tasks_dataloaders)}")

            self.train_loader = train_loader
            self.test_loader = test_loader
            
            self.callback_handler.on_task_begin(
                training_config=self.config, task_id=task_id + 1
            )

            if self.setting == "taskIL":
                self.model.task_id = task_id

            if task_id == 0:
                self.test_loader_first_task = test_loader

            for epoch in range(1, self.config.epochs + 1):
                self.callback_handler.on_epoch_begin(
                    training_config=self.config, 
                    epoch=epoch, 
                    train_loader=self.train_loader, 
                    test_loader=self.test_loader
                )

                epoch_train_loss = self._train_step(epoch)
                metrics.epoch_train_loss = epoch_train_loss

                if self.test_loader is not None:
                    epoch_test_loss, accuracy = self._test_step(epoch, task_id)
                    metrics.epoch_test_loss = epoch_test_loss
                    metrics.accuracy = accuracy

                self.callback_handler.on_epoch_end(training_config=self.config)
                self.callback_handler.on_log(
                    self.config,
                    metrics,
                    logger=logger,
                    epoch=epoch,
                )

            # Test on all seen tasks
            self._test_seen_tasks(task_id)
            self.callback_handler.on_task_end(
                training_config=self.config, task_id=task_id + 1
            )

            if isinstance(self.model, FisherInterface):
                self.model.complete_task(train_loader)
            self._set_optimizer()
            
        if self.save:
            self._save_model()

    def _test_seen_tasks(self, current_task_id):
        for task_id in range(current_task_id + 1):
            logger.info(f"Testing on Task {task_id + 1}/{current_task_id + 1}")
            self.test_loader = self.tasks_dataloaders[task_id][1]

            self.callback_handler.on_test_step_begin(
                training_config=self.config,
                test_loader=self.test_loader,
                epoch=task_id,
            )

            epoch_test_loss, accuracy = self._test_step(0, task_id)
            
            logger.info(
                f"Task {task_id + 1} - Loss: {epoch_test_loss:.4f}, Accuracy: {accuracy:.4f}"
            )


class WandBTrainerCL(TrainerCL):
    def __init__(self, model, tasks_dataloaders, config):
        super().__init__(model, tasks_dataloaders, config)
        self.task_accuracies = []
        self.global_step = 0  # Initialize a global step counter

    def _log_metrics(self, metrics: Dict[str, float], step: int, task_id: Optional[int] = None):
        """Log metrics to WandB."""
        if wandb.run is not None:
            if task_id is not None:
                # Prefix keys with the task id.
                metrics = {f"task_{task_id}/{k}": v for k, v in metrics.items()}
            wandb.log(metrics, step=step)

    def _train_step(self, epoch: int) -> float:
        """Single training step with WandB logging."""
        # Perform training step (callbacks still receive the local epoch if needed)
        epoch_loss = super()._train_step(epoch)
        # Log training loss using the global step counter
        self._log_metrics({"train/loss": epoch_loss}, self.global_step)
        self.global_step += 1  # Increment the global step after each training epoch
        return epoch_loss

    def _test_step(self, step: int, task_id: int = None) -> tuple:
        """Single test step with WandB logging."""
        epoch_loss, accuracy = super()._test_step(step, task_id)
        self._log_metrics({
            "test/loss": epoch_loss,
            "test/accuracy": accuracy
        }, step)
        return epoch_loss, accuracy

    def _test_seen_tasks(self, current_task_id: int):
        """Test on all seen tasks with WandB logging."""
        task_accuracies = []
        for task_id in range(current_task_id + 1):
            logger.info(f"Testing on Task {task_id + 1}/{current_task_id + 1}")
            self.test_loader = self.tasks_dataloaders[task_id][1]

            self.callback_handler.on_test_step_begin(
                training_config=self.config,
                test_loader=self.test_loader,
                epoch=task_id,  # used for callbacks; not for logging step
            )

            # Use the current global_step for testing logging
            epoch_test_loss, accuracy = self._test_step(self.global_step, task_id)
            task_accuracies.append(accuracy)

            # Log per-task test metrics using the current global step
            self._log_metrics({
                "loss": epoch_test_loss,
                "accuracy": accuracy
            }, step=self.global_step, task_id=task_id)

        # Log aggregated metrics for all seen tasks using the current global step
        avg_accuracy = sum(task_accuracies) / len(task_accuracies)
        self._log_metrics({
            "metrics/avg_accuracy": avg_accuracy,
            "metrics/forgetting": max(task_accuracies) - min(task_accuracies)
        }, step=self.global_step)
        self.task_accuracies.append(task_accuracies)

    def train(self):
        """Training loop with WandB logging."""
        # Convert OmegaConf to dict for wandb.
        if wandb.run is not None:
            config_dict = OmegaConf.to_container(self.config, resolve=True)
            wandb.config.update(config_dict)

        super().train()

        # Log final metrics.
        if wandb.run is not None:
            wandb.run.summary.update({
                "final_avg_accuracy": sum(self.task_accuracies[-1]) / len(self.task_accuracies[-1]),
                "final_forgetting": max(self.task_accuracies[0]) - min(self.task_accuracies[-1])
            })

