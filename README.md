# Equilibrium Fisher Control 

## Getting Started

This project uses:
- **Python:** 3.11.9  
- **PyTorch:** 2.6.0  
- **torchvision:** 0.21.0  

Please follow the instructions below to install the required dependencies, initialize WandB, run hyperparameter sweeps, and launch agents across multiple GPUs.

---

So its the EFC_BP_Network, and what you will see is that there is no more alpha_di, no more tau, and no more target_lr.

Now we have lr=0.001 for comparison purposes and to ensure no underfitting. We need to sweep over psi_lr, beta, eps (which now should be no less than 1e-3)

## Installation

1. **Set Up a Virtual Environment:**  
   Create a Python virtual environment called `venv`:
   ```bash
   python -m venv venv
   ```

2. **Activate the Virtual Environment:**
    For Bash:
    ```
    source venv/bin/activate
    ```

    For C-shell (csh):
    ```
    source venv/bin/activate.csh
    ```

3. **Install Required Libraries:**
    With the virtual environment activated, run:

    ```
    pip install -r requirements.txt
    ```

## Setting Up WandB
1. **Create a WandB Account:**
    If you haven't already, sign up for Weights & Biases.

2. **Log In via CLI:**
    Once you have your API key, run:

    ```
    wandb login
    ```
    and paste your API key when prompted.

## Launching a Sweep
This project supports hyperparameter sweeps via WandB. We use CLI arguments (via argparse) to define hyperparameters, and the sweep configuration will override these defaults.
1. **Prepare Your Sweep Config YAML:**
    For example, create a YAML config file  that specifies your hyperparameter ranges (for reference, check `configs/sweep_grid.yaml`).

2. **Initialize the Sweep:**
    Run the following command (replace <project-name> with your WandB project name):

    ```
    wandb sweep --project <project-name> configs/sweep_grid.yaml
    ```
    This command will output a sweep ID (formatted as entity/project/sweep_ID).

## Launching Agents Across Multiple GPUs
To run multiple sweep agents on different GPUs, use the provided start_processes_on_gpu.py script. This script will:

- Check the available VRAM on the specified GPUs.
- Cycle through the GPUs in a round-robin fashion, assigning agents only to those with sufficient free VRAM.

Launch agents using the command:
```
wandb agent <sweep_id>
```

Gracefully cancel the sweep when you press Ctrl+C by triggering:

```
wandb sweep --cancel <sweep_id>
```

1. **Example Usage**:
    To launch 5 agents on GPUs 0, 1, and 2 with a minimum free VRAM threshold of 2500 MiB, run:

    ```
    python start_processes_on_gpu.py <sweep_id> 5 0 1 2 --min_free_vram 2500
    ```

    Replace <sweep_id> with the sweep ID returned from the previous step.

    When you press Ctrl+C, the script will terminate the launched agents and automatically cancel the sweep.

2. **Running a Single Training Run (for Debugging or Local Experiments)**:
    If you want to run a single training job (outside of a sweep), execute:

    ```
    python train.py --layers 784 400 400 2 --lr 1.5e-6 --batch_size 128 --epochs 20 --optimizer Adam --scheduler CosineAnnealingLR --device cuda --seed 1337 --beta_efc 5.0 --target_lr 0.01 --alpha_di 1e-4 --importance_ewc 1.0 --method efc --dt_di 0.008 --time_constant_ratio 0.2 --tmax_di 500 --k_p 2.0 --eps 1e-4 --save false
    ```

    Any parameters not provided will use the default values defined in train.py.

