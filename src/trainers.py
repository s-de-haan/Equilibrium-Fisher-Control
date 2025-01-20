import datetime
import json
import torch
import os
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


class Trainer:
    def __init__(self, model, train_loader, test_loader, config, callbacks=None) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.callbacks = callbacks
        self.config = config
        self.device = self.config.device
        self.loss_fn = self.config.loss_fn

        self._set_device(self.config.device)

        model.to(self.device)

    def train(self) -> None:
        self._prepare_training()
        self.callback_handler.on_train_begin(training_config=self.config)

        logger.info(
            msg=f"Training:\n - epochs: {self.config.epochs}\n - batch_size: {self.config.batch_size}\n - optimizer: {self.config.optimizer}\n - scheduler: {self.config.scheduler}\n - device: {self.config.device}\n - output_dir: {self.config.output_dir}\n - seed: {self.config.seed}\n - encoder_layers: {self.config.encoder_layers}\n - decoder_layers: {self.config.decoder_layers}\n - learning_rate: {self.config.lr}\n - gamma: {self.config.gamma}\n - patience: {self.config.patience}\n - num_workers: {self.config.num_workers}\n - training_dir: {self.training_dir}\n - model: {self.model.name}\n"
        )

        # TODO log to output dir with get_file_logger
        best_train_loss = 1e10
        best_test_loss = 1e10

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
                epoch_test_loss = self._test_step(epoch)
                metrics.epoch_test_loss = epoch_test_loss

            if epoch_test_loss < best_test_loss:
                best_test_loss = epoch_test_loss
                best_model = deepcopy(self.model)
                self._best_model = best_model
            if epoch_train_loss < best_train_loss:
                best_train_loss = epoch_train_loss

            self.callback_handler.on_epoch_end(training_config=self.config)
            self.callback_handler.on_log(
                self.config,
                metrics,
                logger=logger,
                epoch=epoch,
            )

        self._save_model(best_model, dir_path=self.training_dir)
        logger.info(
            f"\nBest train loss: {best_train_loss}, Best test loss: {best_test_loss}"
        )

    def _train_step(self, epoch: int):
        """The trainer performs training loop over the train_loader.

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The step training loss
        """
        self.callback_handler.on_train_step_begin(
            training_config=self.config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        self.model.train()

        epoch_loss = 0

        for (X, y) in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            y_hat = self.model(X)

            loss = self.loss_fn(y_hat, y)

            self.optimizer.zero_grad()
            self.model.set_targets(y)
            self.model.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in train loss")

            self.callback_handler.on_train_step_end(training_config=self.config)

        epoch_loss /= len(self.train_loader)

        return epoch_loss

    @torch.no_grad()
    def _test_step(self, epoch: int):
        """Perform an testuation step

        Parameters:
            epoch (int): The current epoch number

        Returns:
            (torch.Tensor): The testuation loss
        """

        self.callback_handler.on_test_step_begin(
            training_config=self.config,
            test_loader=self.test_loader,
            epoch=epoch,
        )

        epoch_loss = 0
        total = 0
        correct = 0

        for (X, y) in self.test_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            with torch.no_grad():
                y_hat = self.model(X)

            loss = self.loss_fn(y_hat, y)

            epoch_loss += loss.item()

            _, predicted = torch.max(y_hat.data, 1)
            total += y.size(0)
            correct += (predicted == y.argmax(dim=1)).sum().item()

            if epoch_loss != epoch_loss:
                raise ArithmeticError("NaN detected in test loss")

            self.callback_handler.on_test_step_end(training_config=self.config)

        epoch_loss /= len(self.test_loader)
        accuracy = 100 * correct / total
        print(f"Accuracy: {accuracy}")

        return epoch_loss

    def _set_device(self, device: str):
        self.device = torch.device(device)
        torch.set_default_device(self.device)

    def _save_model(self, model, dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        torch.save(model.state_dict(), os.path.join(dir_path, "model.pt"))

        with open(os.path.join(dir_path, "config.json"), "w") as fp:
            json.dump(self.config, fp)

        self.callback_handler.on_save(self.config)

    def _setup_logger(self):
        logging.basicConfig(
            filename=os.path.join(self.training_dir, "training.log"),
            level=logging.INFO,
            format="%(message)s",
        )

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
