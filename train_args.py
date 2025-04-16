import argparse
from src.utils import str2bool

def parse_args():
    parser = argparse.ArgumentParser(description="Train continual learning model using CLI args.")

    # Network architecture & training hyperparameters:
    parser.add_argument("--layers", type=int, nargs='+', default=[784, 400, 400, 10],
                        help="Network layer sizes (e.g., 784 400 400 2)")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--mode", type=str, default="di", choices=["ndi", "di"],
                        help="whether to run with (di) or without (ndi) dynamic inversion")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for dataloader")
    parser.add_argument("--loss_fn", type=str, default='ce',
                        help="whether to train with cross entropy ('ce') or mean squared error ('mse') loss")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD"], help="Optimizer")
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR", help="Scheduler")

    # Environment settings
    parser.add_argument("--device", type=str, default="cuda", help="GPU/CPU device")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--save", type=str2bool, default="false", help="Whether to save the model")

    # EFC-specific
    parser.add_argument("--clamp", type=str2bool, default="false", help="Whether to clamp")
    parser.add_argument("--lr", type=float, default=1.5e-6, help="Learning rate")
    parser.add_argument("--beta_efc", type=float, default=5000.0)
    parser.add_argument("--target_lr", type=float, default=1.0)
    parser.add_argument("--alpha_di", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.008)
    parser.add_argument("--eps", type=float, default=1e-4)

    # EWC-specific
    parser.add_argument("--importance_ewc", type=float, default=1.0)

    # Method selection (now includes DynDFC)
    parser.add_argument("--method", type=str, default="efc", choices=["efc", "bp", "efc_bp", "dyn_dfc"],
                        help="Training method to use")

    # DynDFC-specific
    parser.add_argument("--eta_dyn", type=float, default=0.1, help="Learning rate for feedback weights W_dyn")
    parser.add_argument("--eta_ff", type=float, default=1e-3, help="Learning rate for feedforward weights")
    parser.add_argument("--k_p", type=float, default=2.0, help="Proportional gain for dynamic inversion")
    parser.add_argument("--k_i", type=float, default=0.0, help="Integral gain for dynamic inversion")
    parser.add_argument("--k_d", type=float, default=0.0, help="Derivative gain for dynamic inversion")

    # Dynamic inversion control
    parser.add_argument("--dt_di", type=float, default=0.008)
    parser.add_argument("--time_constant_ratio", type=float, default=0.2)
    parser.add_argument("--tmax_di", type=int, default=500)

    # Other
    parser.add_argument("--run_name", type=str, default=None, help="Name of the run")
    parser.add_argument("--psi_lr", type=float, default=0.5)
    parser.add_argument("--setting", type=str, default="domainIL", choices=["taskIL", "classIL", "domainIL"])

    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unknown CLI arguments:", unknown)
    return args
