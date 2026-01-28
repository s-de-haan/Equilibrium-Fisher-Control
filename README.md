# Equilibrium Fisher Control 

## Getting Started

This project uses:
- **Python:** 3.11.9  
- **PyTorch:** 2.6.0  
- **torchvision:** 0.21.0  

Please follow the instructions below to install the required dependencies, initialize WandB, run hyperparameter sweeps, and launch agents across multiple GPUs. We also provide a script to run any of the results we present in the table. 

---

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


## Reproduce Paper Results 

Every hyperparameters configuration of all models & training setting presented in the paper are in the `./final_configs` folder, organized by method. These config files are however *"wandb sweep"* config files, so in case you'd want to run a single one of them, please use the `./final_configs/single_run_template.yaml` template (see 1.). Alternatively, if you'd like to reproduce any result accross 5 seeds as we do in the paper, follow 2.,  

1. **Run a single model with specific hyperparameters**

`WANDB_MODE=disabled python train.py --config configs/single_run_template.yaml`

To enable WANDB, simply run with `WANDB_MODE=enabled`. 

2. **Reproduce results accross 5 seeds**

`python start_processes_on_gpu.py --config final_configs/<method>/<setting>.yaml`

If you have multiple (e.g. 4) GPUs, you can parallelize a run as such: 

`python start_processes_on_gpu.py 0 1 2 3 --config final_configs/<method>/<setting>.yaml`

---

## Training Pipeline Structure

```
train.py                     # Entry point: CLI parsing, model/dataset selection, launches trainer
├── src/
│   ├── trainers.py          # TrainerCL / WandBTrainerCL: continual learning loop
│   ├── dataloaders_2.py     # TaskIL/ClassIL dataloaders with optional CNN encoder
│   ├── callbacks.py         # Progress bars, logging hooks
│   └── config.py            # OmegaConf-based configuration (unused in main flow)
└── networks/
    ├── network_interface.py # Base Network class, FisherInterface, JacobianInterface
    ├── EFC_network.py       # Equilibrium Fisher Control (our method)
    ├── EWC_network.py       # Elastic Weight Consolidation baseline
    ├── oEWC_network.py      # Online EWC baseline
    ├── SI_network.py        # Synaptic Intelligence baseline
    ├── BP_network.py        # Standard backprop (no regularization)
    └── layers.py            # Custom layer implementations with backward hooks
```

### Flow

1. **Configuration**: CLI args + optional YAML config → `OmegaConf` object
2. **Model**: `get_model()` instantiates network based on `--method` (efc, ewc, oewc, si, bp)
3. **Data**: `get_dataset()` returns a list of `(train_loader, test_loader)` per task
4. **Training**: `WandBTrainerCL.train()` loops over tasks:
   - For each task: trains for `epochs`, evaluates on all seen tasks
   - Calls `model.complete_task()` to update Fisher/importance weights
   - Resets optimizer and scheduler between tasks

### Key Abstractions

- **`FisherInterface`**: Computes diagonal Fisher after each task, stores θ* (optimal params)
- **`JacobianInterface`**: Implements dynamic/non-dynamic target inversion for EFC
- **Task masks**: `Network._setup_task_masks()` handles Class-IL vs Task-IL output masking
