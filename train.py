import argparse
import wandb
from omegaconf import OmegaConf
import torch
from torch.utils.data import ConcatDataset, DataLoader

from networks.DynDFC_network import DynDFC_network
from src.datasets import SplitMNIST
from src.trainers import TrainerCL
from src.utils import str2bool


def parse_args():
    parser = argparse.ArgumentParser(description="Class-incremental learning with DynDFC")
    # Model and training parameters
    parser.add_argument('--layers', type=int, nargs='+', default=[784, 400, 400, 2],
                        help='List of layer sizes')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of output classes')
    parser.add_argument('--lr', type=float, default=1e-3, help='Base learning rate')
    parser.add_argument('--eta_ff', type=float, default=1e-3,
                        help='Learning rate for feedforward weights')
    parser.add_argument('--eta_dyn', type=float, default=0.1,
                        help='Learning rate for feedback weights')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=4, help='Number of epochs')
    parser.add_argument('--mode', type=str, default='di', choices=['di', 'ndi'],
                        help='Dynamic inversion mode')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers')

    # Loss and optimizer
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'mse'],
                        help='Loss function')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help='Optimizer')
    parser.add_argument('--scheduler', type=str, default='CosineAnnealingLR',
                        help='Learning rate scheduler')

    # Controller parameters
    parser.add_argument('--dt_di', type=float, default=0.0016,
                        help='Time step for dynamic inversion')
    parser.add_argument('--time_constant_ratio', type=float, default=0.2,
                        help='Time constant ratio for dynamics')
    parser.add_argument('--tmax_di', type=int, default=500,
                        help='Max inference steps')
    parser.add_argument('--k_p', type=float, default=1.0, help='Proportional gain')
    parser.add_argument('--k_i', type=float, default=0.0, help='Integral gain')
    parser.add_argument('--k_d', type=float, default=0.0, help='Derivative gain')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Convergence threshold')
    parser.add_argument('--alpha_di', type=float, default=1e-4,
                        help='Controller damping term')
    parser.add_argument('--target_lr', type=float, default=1e-2,
                        help='Target learning rate for control targets')

    # Environment
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save', type=str2bool, default='false', help='Save model')

    # Continual-learning setting (unused for random sampling)
    parser.add_argument('--setting', type=str, default='classIL', help='Continual learning setting')
    parser.add_argument('--dataset', type=str, default='MNIST', help='Dataset for CL')

    args, unknown = parser.parse_known_args()
    if unknown:
        print('Ignoring unknown args:', unknown)
    return args


def main():
    args = parse_args()
    # Incorporate WandB sweep parameters
    if wandb.run is not None:
        for k, v in wandb.config.items():
            setattr(args, k, v)

    # Build configuration
    config = OmegaConf.create(vars(args))

    # Initialize WandB
    wandb.init(
        project='classIL_DynDFC',
        config=config,
        reinit=True
    )

    # Load all task splits, then concatenate into a single dataset
    raw_tasks = SplitMNIST(config=config).get_all_tasks_dataloaders()
    train_datasets = [task[0].dataset for task in raw_tasks]
    test_datasets  = [task[1].dataset for task in raw_tasks]

    train_all = ConcatDataset(train_datasets)
    test_all  = ConcatDataset(test_datasets)

    train_loader_all = DataLoader(
        train_all,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
    test_loader_all = DataLoader(
        test_all,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )

    # Wrap into tasks list for TrainerCL compatibility
    tasks = [(train_loader_all, test_loader_all)]

    # Instantiate DynDFC model
    model = DynDFC_network(config=config)

    # Train
    trainer = TrainerCL(model, tasks, config)
    trainer.train()


if __name__ == '__main__':
    main()
