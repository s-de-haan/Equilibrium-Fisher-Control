import datetime
import json
import torch
import os
import sys
import logging

from copy import deepcopy
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
        self.loss_fn = config.loss_fn
        self.save = config.save
        self.callbacks = callbacks

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
                self.model.parameters(), lr=self.config["lr"]
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
        elif self.config.scheduler == None:
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

    def train_step(self, batch):
        raise NotImplementedError("train_step must be implemented in a subclass.")

    def test_step(self, batch):
        raise NotImplementedError("test_step must be implemented in a subclass.")

    def train(self):
        raise NotImplementedError("train must be implemented in a subclass.")


class Trainer(TrainerInterface):
    def __init__(
        self, model, train_loader, test_loader, config, callbacks=None
    ) -> None:
        super().__init__(model, config, callbacks)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self) -> None:
        self.model.zero_grad()

        for epoch in range(1, self.config.epochs + 1):
            self.callback_handler.on_epoch_begin(
                training_config=self.config,
                epoch=epoch,
                train_loader=self.train_loader,
                test_loader=self.test_loader,
            )

            metrics = dotdict()

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

            loss = self.loss_fn(y_hat, y)
            # TODO: test
            # Jis_torch = torch.autograd.functional.jacobian(self.model, X)
            
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
    def _test_step(self, epoch: int):
        self.callback_handler.on_test_step_begin(
            training_config=self.config,
            test_loader=self.test_loader,
            epoch=epoch,
        )

        epoch_loss = 0
        total = 0
        correct = 0

        for X, y in self.test_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_hat = self.model(X)

            loss = self.loss_fn(y_hat, y)

            epoch_loss += loss.item()
            total += y.size(0)
            correct += (y_hat.argmax(dim=1) == y.argmax(dim=1)).sum().item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in test loss")
            self.callback_handler.on_test_step_end(training_config=self.config)

        epoch_loss /= len(self.test_loader)
        accuracy = 100 * correct / total

        return epoch_loss, accuracy


class TrainerCL(TrainerInterface):
    def __init__(self, model, tasks_dataloaders, config, callbacks=None):
        super().__init__(model, config, callbacks)
        self.tasks_dataloaders = tasks_dataloaders
        self.task_results = []

    def train(self):
        self.callback_handler.on_train_begin(training_config=self.config)

        for task_id, (train_loader, _) in enumerate(self.tasks_dataloaders):
            logger.info(f"Starting Task {task_id + 1}/{len(self.tasks_dataloaders)}")
            self.callback_handler.on_task_begin(
                training_config=self.config, task_id=task_id + 1
            )

            for epoch in range(self.config.epochs):
                self.callback_handler.on_epoch_begin(
                    training_config=self.config, epoch=epoch + 1
                )
                train_loss = 0.0

                self.callback_handler.on_train_step_begin(
                    training_config=self.config, train_loader=train_loader, epoch=epoch
                )
                for batch in train_loader:
                    loss = self._train_step(batch)
                    train_loss += loss

                train_loss /= len(train_loader)
                logger.info(
                    f"Task {task_id + 1}, Epoch {epoch + 1}/{self.config.epochs}, Loss: {train_loss:.4f}"
                )
                self.callback_handler.on_epoch_end(
                    training_config=self.config, epoch=epoch + 1, loss=train_loss
                )

            # Test on all seen tasks
            self._test_seen_tasks(task_id)
            self.callback_handler.on_task_end(
                training_config=self.config, task_id=task_id + 1
            )
        
        if self.save:
            self._save_model()


    def _train_step(self, batch):
        self.model.train()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self.optimizer.step()

        self.callback_handler.on_train_step_end(training_config=self.config)

        return loss.item()

    def _test_step(self, batch):
        self.model.eval()
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        with torch.no_grad():
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            accuracy = (
                (torch.argmax(outputs, dim=1) == torch.argmax(targets, dim=1))
                .float()
                .mean()
                .item()
            )

        return loss.item(), accuracy

    def _test_seen_tasks(self, current_task_id):
        results = {}
        for task_id in range(current_task_id + 1):
            logger.info(f"Testing on Task {task_id + 1}/{current_task_id + 1}")
            test_loader = self.tasks_dataloaders[task_id][1]
            self.test_loader = test_loader

            self.callback_handler.on_test_step_begin(
                training_config=self.config,
                test_loader=self.test_loader,
                epoch=task_id,
            )

            test_loss, test_acc = 0.0, 0.0
            for batch in test_loader:
                loss, acc = self._test_step(batch)
                test_loss += loss
                test_acc += acc

                self.callback_handler.on_test_step_end(training_config=self.config)

            test_loss /= len(test_loader)
            test_acc /= len(test_loader)
            logger.info(
                f"Task {task_id + 1} - Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}"
            )

            results[f"Task {task_id + 1}"] = {"loss": test_loss, "accuracy": test_acc}

        self.task_results.append(deepcopy(results))
