import os
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from networks.GAN_network import Discriminator, Generator, labels_to_onehot
from networks.network_interface import FisherInterface
from src.trainers import TrainerCL, logger
from src.utils import dotdict


class TrainerGAN:
    """
    Trainer for the conditional MNIST GAN used for replay.

    This is the notebook GAN training loop packaged into a reusable class.
    It accepts either:
    - a dataloader yielding ``(image, class_index)``, or
    - a dataloader yielding ``(image, one_hot_target)``.
    """

    def __init__(
        self,
        config,
        train_loader,
        generator=None,
        discriminator=None,
        reinitialize=True,
    ):
        self.config = config
        self.train_loader = train_loader
        self.device = torch.device(config.get("device", "cpu"))

        self.latent_dim = config.get("gan_latent_dim", 100)
        self.num_classes = (
            generator.num_classes if generator is not None
            else config.get("gan_num_classes", 10)
        )
        self.hidden_dim = config.get("gan_hidden_dim", 200)
        self.image_dim = config.get("gan_image_dim", 784)
        self.gan_lr = config.get("gan_lr", 2e-4)
        self.gan_epochs = config.get("gan_epochs", 50)
        self.result_root = config.get("gan_result_root", "MNIST_cGAN_results")
        self.gan_generator_path = config.get(
            "gan_generator_path", os.path.join(self.result_root, "generator.pth")
        )
        self.gan_discriminator_path = config.get(
            "gan_discriminator_path", os.path.join(self.result_root, "discriminator.pth")
        )
        self.gan_lr_decay_epochs = config.get("gan_lr_decay_epochs", [29, 39])

        os.makedirs(self.result_root, exist_ok=True)

        self.generator = generator or Generator(
            latent_dim=self.latent_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            image_dim=self.image_dim,
        )
        self.discriminator = discriminator or Discriminator(
            image_dim=self.image_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
        )

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        if reinitialize:
            self.generator.weight_init(0, 0.02)
            self.discriminator.weight_init(0, 0.02)

        self.criterion = nn.BCELoss()
        self.g_optimizer = optim.Adam(
            self.generator.parameters(), lr=self.gan_lr, betas=(0.5, 0.999)
        )
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.gan_lr, betas=(0.5, 0.999)
        )

        self.train_hist = {"D_losses": [], "G_losses": [], "per_epoch_ptimes": []}

    def _prepare_real_batch(self, x_real, y_real):
        x_real = x_real.to(self.device, dtype=torch.float32)
        x_real = x_real.view(x_real.size(0), -1)

        if y_real.dim() > 1 and y_real.size(-1) == self.num_classes:
            y_real_onehot = y_real.to(self.device, dtype=torch.float32)
        else:
            y_real_onehot = labels_to_onehot(
                y_real.to(self.device, dtype=torch.long), self.num_classes
            ).to(dtype=x_real.dtype)

        return x_real, y_real_onehot

    def _sample_fake_labels(self, batch_size):
        labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)
        return labels_to_onehot(labels, self.num_classes).to(dtype=torch.float32)

    def save(self):
        torch.save(self.generator.state_dict(), self.gan_generator_path)
        torch.save(self.discriminator.state_dict(), self.gan_discriminator_path)

    def train(self, epochs=None, train_loader=None, save=True):
        if train_loader is not None:
            self.train_loader = train_loader

        num_epochs = self.gan_epochs if epochs is None else epochs

        print("GAN training start!")
        start_time = time.time()

        for epoch in range(num_epochs):
            d_losses, g_losses = [], []

            """if epoch in self.gan_lr_decay_epochs:
                self.g_optimizer.param_groups[0]["lr"] /= 10
                self.d_optimizer.param_groups[0]["lr"] /= 10
                print("GAN learning rate change!")"""

            epoch_start_time = time.time()

            for x_real, y_real in self.train_loader:
                x_real, y_real_onehot = self._prepare_real_batch(x_real, y_real)
                mini_batch = x_real.size(0)

                real_target = torch.ones(mini_batch, device=self.device)
                fake_target = torch.zeros(mini_batch, device=self.device)

                self.discriminator.zero_grad()

                d_real = self.discriminator(x_real, y_real_onehot).view(-1)
                d_real_loss = self.criterion(d_real, real_target)

                z = torch.rand(mini_batch, self.latent_dim, device=self.device)
                y_fake_onehot = self._sample_fake_labels(mini_batch)
                fake_images = self.generator(z, y_fake_onehot)
                d_fake = self.discriminator(fake_images.detach(), y_fake_onehot).view(-1)
                d_fake_loss = self.criterion(d_fake, fake_target)

                d_loss = d_real_loss + d_fake_loss
                d_loss.backward()
                self.d_optimizer.step()

                self.generator.zero_grad()

                z = torch.rand(mini_batch, self.latent_dim, device=self.device)
                y_fake_onehot = self._sample_fake_labels(mini_batch)
                fake_images = self.generator(z, y_fake_onehot)
                d_fake = self.discriminator(fake_images, y_fake_onehot).view(-1)

                g_loss = self.criterion(d_fake, real_target)
                g_loss.backward()
                self.g_optimizer.step()

                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())

            epoch_time = time.time() - epoch_start_time
            mean_d_loss = float(np.mean(d_losses)) if d_losses else float("nan")
            mean_g_loss = float(np.mean(g_losses)) if g_losses else float("nan")

            self.train_hist["D_losses"].append(mean_d_loss)
            self.train_hist["G_losses"].append(mean_g_loss)
            self.train_hist["per_epoch_ptimes"].append(epoch_time)

            print(
                f"[{epoch + 1}/{num_epochs}] "
                f"D: {mean_d_loss:.3f}, G: {mean_g_loss:.3f}"
            )

        total_time = time.time() - start_time
        print(f"GAN training finished in {total_time:.2f} seconds")
        if save:
            self.save()

        return self.train_hist


class TrainerCL_DFC_GAN(TrainerCL):
    """
    DFC continual-learning trainer with GAN-based replay.

    This mirrors ``TrainerCL_DFC`` from ``src/trainers.py``, but replaces the
    replay sample generation step with a pretrained conditional GAN generator.
    """

    def __init__(
        self,
        model,
        tasks_dataloaders,
        config,
        callbacks=None,
    ):
        super().__init__(model, tasks_dataloaders, config, callbacks)

        self.recon_loss_fn = torch.nn.MSELoss()

        self.total_num_classes = config.get("gan_num_classes", 10)
        self.bzs_gen = config.get("batch_size_gen", config.batch_size)
        self.flatten_imgs = (
            True if config.get("flatten_imgs", "default") == "default"
            else str(config.get("flatten_imgs")).lower() == "true"
        )

        self.gen_samples = []
        self.gen_targets = []
        self.gen_datasets = []

        self.gan_generator_path = config.get(
            "gan_generator_path", "MNIST_cGAN_results/generator.pth"
        )
        self.gan_discriminator_path = config.get(
            "gan_discriminator_path", "MNIST_cGAN_results/discriminator.pth"
        )
        self.gan_latent_dim = config.get("gan_latent_dim", 100)
        self.gan_hidden_dim = config.get("gan_hidden_dim", 200)
        self.gan_image_dim = config.get("gan_image_dim", 784)
        self.data_mean = config.get("data_mean", 0.1307)
        self.data_std = config.get("data_std", 0.3081)
        self.gan_retrain_epochs = config.get("gan_retrain_epochs", 5)
        self.current_gan_num_classes = 0
        self.gan = None
        self.gan_discriminator = None
        self.replay_rng = self._build_replay_rng()
        self.gan_trainer = None

        self._set_dfcl_optimizers()

    def _build_replay_rng(self):
        try:
            return torch.Generator(device=self.device).manual_seed(self.config.seed)
        except RuntimeError:
            return torch.Generator(device="cpu").manual_seed(self.config.seed)

    def _load_generator(self):
        if not os.path.exists(self.gan_generator_path):
            raise FileNotFoundError(
                f"GAN generator checkpoint not found: {self.gan_generator_path}"
            )

        generator = Generator(
            latent_dim=self.gan_latent_dim,
            num_classes=self.total_num_classes,
            hidden_dim=self.gan_hidden_dim,
            image_dim=self.gan_image_dim,
        ).to(self.device)

        state_dict = torch.load(self.gan_generator_path, map_location=self.device)
        generator.load_state_dict(state_dict)
        generator.eval()
        return generator

    def _load_discriminator(self):
        discriminator = Discriminator(
            image_dim=self.gan_image_dim,
            num_classes=self.total_num_classes,
            hidden_dim=self.gan_hidden_dim,
        ).to(self.device)

        if os.path.exists(self.gan_discriminator_path):
            state_dict = torch.load(self.gan_discriminator_path, map_location=self.device)
            discriminator.load_state_dict(state_dict)
        else:
            discriminator.weight_init(0, 0.02)

        discriminator.eval()
        return discriminator

    def _num_seen_classes(self, task_id):
        return min(
            (task_id + 1) * self.config["classes_per_task"],
            self.total_num_classes,
        )

    def _create_new_gan(self, task_id):
        gan_num_classes = self._num_seen_classes(task_id)
        generator = Generator(
            latent_dim=self.gan_latent_dim,
            num_classes=gan_num_classes,
            hidden_dim=self.gan_hidden_dim,
            image_dim=self.gan_image_dim,
        ).to(self.device)
        discriminator = Discriminator(
            image_dim=self.gan_image_dim,
            num_classes=gan_num_classes,
            hidden_dim=self.gan_hidden_dim,
        ).to(self.device)

        generator.weight_init(0, 0.02)
        discriminator.weight_init(0, 0.02)

        self.gan = generator
        self.gan_discriminator = discriminator
        self.current_gan_num_classes = gan_num_classes
        gan_config = dotdict(dict(self.config))
        gan_config["gan_num_classes"] = gan_num_classes
        self.gan_trainer = TrainerGAN(
            config=gan_config,
            train_loader=None,
            generator=self.gan,
            discriminator=self.gan_discriminator,
            reinitialize=False,
        )
        self.gan.eval()
        self.gan_discriminator.eval()

    def _scale_generated_images(self, images):
        """
        Map GAN outputs from the GAN training range to the continual-learning
        dataloader range.

        The GAN was trained on MNIST normalized with mean/std = (0.5, 0.5),
        so its tanh outputs are in [-1, 1]. The CL dataloaders normalize MNIST
        with mean/std = (0.1307, 0.3081), so replay images must be converted to
        [0, 1] first and then normalized with those statistics.
        """
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)
        return (images - self.data_mean) / self.data_std

    def _convert_cl_images_to_gan_range(self, images):
        """
        Convert images from continual-learning normalization back to the GAN
        training normalization.

        CL range:
            x_cl = (x - data_mean) / data_std

        GAN range:
            x_gan = 2 * x - 1
        """
        images = images * self.data_std + self.data_mean
        images = torch.clamp(images, 0.0, 1.0)
        return images * 2.0 - 1.0

    def _sample_labels_from_seen_tasks(self, current_task_id, num_samples):
        num_seen_classes = min(
            (current_task_id + 1) * self.config["classes_per_task"],
            self.total_num_classes,
        )
        return torch.randint(
            low=0,
            high=num_seen_classes,
            size=(num_samples,),
            generator=self.replay_rng,
        )

    def _sample_labels_from_previous_tasks(self, current_task_id, num_samples):
        num_prev_classes = min(
            current_task_id * self.config["classes_per_task"],
            self.total_num_classes,
        )
        if num_prev_classes <= 0:
            return torch.empty(0, dtype=torch.long)

        return torch.randint(
            low=0,
            high=num_prev_classes,
            size=(num_samples,),
            generator=self.replay_rng,
        )

    def _build_gan_retrain_dataset(self, task_id):
        gan_num_classes = self._num_seen_classes(task_id)
        real_x_batches = []
        real_y_batches = []

        for x_real, y_real in self.train_loader:
            x_real = x_real.to(self.device, dtype=torch.float32)
            y_real = y_real.to(self.device, dtype=torch.float32)

            x_real = self._convert_cl_images_to_gan_range(x_real)
            x_real = x_real.view(x_real.shape[0], -1)

            if y_real.dim() == 1:
                y_real = labels_to_onehot(y_real.long(), gan_num_classes).to(
                    dtype=x_real.dtype
                )
            else:
                y_real = y_real[:, :gan_num_classes]

            real_x_batches.append(x_real.detach().cpu())
            real_y_batches.append(y_real.detach().cpu())

        current_real_x = torch.cat(real_x_batches, dim=0)
        current_real_y = torch.cat(real_y_batches, dim=0)

        if task_id == 0:
            logger.info(
                "Training GAN after task 1 using only real data from the first task."
            )
            return TensorDataset(current_real_x, current_real_y)

        num_prev_samples = current_real_x.shape[0]
        prev_labels = self._sample_labels_from_previous_tasks(
            task_id, num_prev_samples
        ).to(self.device)
        prev_onehot = labels_to_onehot(prev_labels, gan_num_classes).to(
            dtype=current_real_x.dtype
        )

        with torch.no_grad():
            prev_images = self.gan.generate_from_labels(
                prev_labels,
                reshape=False,
                generator=self.replay_rng,
            )

        combined_x = torch.cat([current_real_x, prev_images.detach().cpu()], dim=0)
        combined_y = torch.cat([current_real_y, prev_onehot.detach().cpu()], dim=0)
        return TensorDataset(combined_x, combined_y)

    def _retrain_gan_after_task(self, task_id):
        retrain_dataset = self._build_gan_retrain_dataset(task_id)
        if retrain_dataset is None:
            return

        self._create_new_gan(task_id)

        try:
            retrain_generator = torch.Generator(device=self.device).manual_seed(
                self.config.seed + task_id
            )
        except RuntimeError:
            retrain_generator = torch.Generator(device="cpu").manual_seed(
                self.config.seed + task_id
            )

        retrain_loader = DataLoader(
            retrain_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True,
            generator=retrain_generator,
        )

        logger.info(
            f"Training a new GAN after task {task_id + 1} for {self.gan_retrain_epochs} epochs"
        )
        self.gan.train()
        self.gan_discriminator.train()
        self.gan_trainer.train(
            epochs=self.gan_retrain_epochs,
            train_loader=retrain_loader,
            save=True,
        )
        self.gan.eval()
        self.gan_discriminator.eval()

    def _set_dfcl_optimizers(self):
        lr = self.config.lr_fb

        self.opt_fb = torch.optim.Adam(
            self.model.feedback_layers.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=5.832e-07,
        )

        print("[TrainerCL_DFC_GAN] Created optimizers: feedforward, 1x feedback.")

    def _sample_replay_labels(self, num_active_classes):
        labels = []

        for _ in range(self.model.task_id):
            for batch in self.train_loader:
                batch_size = batch[0].shape[0]
                rand = torch.randint(
                    low=0,
                    high=num_active_classes,
                    size=(batch_size,),
                    generator=self.replay_rng,
                )
                labels.append(rand)

        if len(labels) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)

        return torch.cat(labels, dim=0).to(self.device)

    @torch.no_grad()
    def _generate_gan_replay_dataset(self):
        num_active_classes = min(
            self.config["classes_per_task"] * self.model.task_id,
            self.gan.num_classes,
        )

        replay_labels = self._sample_replay_labels(num_active_classes)
        if replay_labels.numel() == 0:
            return

        generated_batches = []
        for start in range(0, replay_labels.shape[0], self.bzs_gen):
            label_batch = replay_labels[start : start + self.bzs_gen]
            batch_images = self.gan.generate_from_labels(
                label_batch,
                reshape=not self.flatten_imgs,
                generator=self.replay_rng,
            )
            batch_images = self._scale_generated_images(batch_images)

            if self.flatten_imgs and batch_images.dim() > 2:
                batch_images = batch_images.view(batch_images.shape[0], -1)

            generated_batches.append(batch_images)

        generated_inputs_full = torch.cat(generated_batches, dim=0).to(self.device)
        target = torch.softmax(self.model(generated_inputs_full), dim=1)

        self.gen_samples.append(generated_inputs_full.detach())
        self.gen_targets.append(target.detach())

        gen_x = generated_inputs_full.detach().cpu()
        gen_y = target.detach().cpu()

        self.gen_datasets = [TensorDataset(gen_x, gen_y)]

    def _train_step(self, epoch: int):
        self.callback_handler.on_train_step_begin(
            training_config=self.config,
            train_loader=self.train_loader,
            epoch=epoch,
        )

        self.model.train()
        if self.gan is not None:
            self.gan.eval()

        epoch_loss = 0.0
        num_updates = 0
        layers = self.model.layers
        self.model.converged_per_batch = []
        num_total_classes = self.model.layers[-1].out_features

        if (self.model.task_id >= 1) and (epoch == 1):
            if self.gan is None:
                raise RuntimeError(
                    "GAN replay requested before a GAN was trained. "
                    "Finish task 1 GAN training before starting replay."
                )
            self._generate_gan_replay_dataset()

        replay_loader = None
        replay_iter = None

        if len(self.gen_datasets) > 0:
            replay_loader = DataLoader(
                self.gen_datasets[0],
                batch_size=self.train_loader.batch_size,
                shuffle=True,
                drop_last=False,
                generator=self._build_replay_rng(),
                num_workers=self.config.num_workers,
                pin_memory=True,
            )
            replay_iter = iter(replay_loader)

        def _run_batch(X, y, update_weight=1.0, replay=False):
            nonlocal epoch_loss, num_updates, num_total_classes

            X = X.to(self.device)
            y = y.to(self.device)

            y_hat = self.model(X)
            primary_loss = self.model.calculate_loss(y_hat, y)

            for j, layer in enumerate(layers):
                layer_fb = self.model.feedback_layers[j]

                if j == len(layers) - 1:
                    if replay:
                        r_i1 = y.argmax(dim=1).clone().detach()
                        r_i1 = F.one_hot(r_i1, num_classes=num_total_classes).float()
                    else:
                        r_i1 = y.clone().detach()
                else:
                    r_i1 = layer.r.clone().detach()

                if j == 0:
                    r_prev = self.model.input.clone().detach()
                    r_rec = layer_fb(r_i1)
                else:
                    r_prev = layers[j - 1].r.clone().detach()
                    r_rec = layers[j - 1].activation_fn(
                        layer_fb(r_i1) + layers[j - 1].bias.unsqueeze(0)
                    )

                fb_recon_loss = self.recon_loss_fn(r_rec, r_prev)
                self.opt_fb.zero_grad()
                fb_recon_loss.backward()
                self.opt_fb.step()

            with torch.no_grad():
                self.model.label = y
                self.optimizer.zero_grad()
                self.model.backward(y)

                if update_weight != 1.0:
                    for p in self.model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(update_weight)

                self.optimizer.step()

            epoch_loss += primary_loss.item() * update_weight
            num_updates += 1
            self.callback_handler.on_train_step_end(training_config=self.config)

        current_weight = 1 / (self.model.task_id + 1)
        replay_weight = 1 - current_weight

        for X_real, y_real in self.train_loader:
            _run_batch(X_real, y_real, update_weight=current_weight)

            if replay_loader is not None:
                try:
                    X_rep, y_rep = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    X_rep, y_rep = next(replay_iter)

                _run_batch(X_rep, y_rep, update_weight=replay_weight, replay=True)

        if len(self.model.converged_per_batch) > 0:
            self.model.mean_convergence_per_epoch.append(
                np.array(self.model.converged_per_batch).mean()
            )
        else:
            self.model.mean_convergence_per_epoch.append(np.nan)

        return epoch_loss / max(num_updates, 1)

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

            self.model.task_id = task_id

            if task_id == 0:
                self.test_loader_first_task = test_loader

            if self.use_peak and task_id > 0:
                self.best_cumulative_accuracy = -float("inf")
                self.best_model_state = None
                self.peak_epoch = 0

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
                    if self.config.setting in ["classIL5task", "classIL2task"]:
                        epoch_test_loss, accuracy = self._test_step(epoch, task_id)
                        task_losses = getattr(self, "current_task_losses", [])
                        task_accuracies = getattr(self, "current_task_accuracies", [])
                        metrics.task_losses = task_losses
                        metrics.task_accuracies = task_accuracies
                        metrics.cumulative_accuracy = accuracy
                        self.accuracies_full.append(accuracy)
                        for t_num, t in enumerate(task_accuracies):
                            self.accuracies_tasks[t_num].append(t)

                        if self.use_peak and task_id > 0:
                            if accuracy > self.best_cumulative_accuracy:
                                self.best_cumulative_accuracy = accuracy
                                self.best_model_state = copy.deepcopy(
                                    self.model.state_dict()
                                )
                                self.peak_epoch = epoch
                                logger.info(
                                    "New peak model saved at epoch "
                                    f"{epoch} with cumulative accuracy: {accuracy:.2f}%"
                                )
                    else:
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

            if self.use_peak and task_id > 0 and self.best_model_state is not None:
                logger.info(
                    "Restoring peak model from epoch "
                    f"{self.peak_epoch} with cumulative accuracy "
                    f"{self.best_cumulative_accuracy:.2f}"
                )
                self.model.load_state_dict(self.best_model_state)

            self._retrain_gan_after_task(task_id)

            self._test_seen_tasks(task_id)
            self.callback_handler.on_task_end(
                training_config=self.config, task_id=task_id + 1
            )

            if isinstance(self.model, FisherInterface):
                self.model.complete_task(train_loader)
            self._set_optimizer()

            if self.first_task_only:
                break

        if self.save:
            self._save_model()


TrainerCL_GAN = TrainerCL_DFC_GAN
