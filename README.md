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

`WANDB_MODE=disabled python train.py --config configs/single_run.yaml`

To enable WANDB, simply run with `WANDD_MODE=enabled`. 

2. **Reproduce results accross 5 seeds**

`python start_processes_on_gpu.py --config final_configs/<method>/<setting>.yaml`

If you have multiple (e.g. 4) GPUs, you can parallelize a run as such: 

`python start_processes_on_gpu.py 0 1 2 3 --config final_configs/<method>/<setting>.yaml`
