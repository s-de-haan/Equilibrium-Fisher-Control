#!/usr/bin/env python3
import argparse
import subprocess
import sys
import time

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Launch multiple WandB agents (wandb agent <sweep_id>) on specified GPUs if enough free VRAM is available."
    )
    parser.add_argument('sweep_id', type=str, help='WandB sweep ID (e.g., entity/project/sweep_id)')
    parser.add_argument('n_processes', type=int, help='Number of agents (processes) to start')
    parser.add_argument('gpu_ids', type=int, nargs='+', help='List of GPU IDs to use')
    parser.add_argument('--min_free_vram', type=int, default=2000,
                        help='Minimum free VRAM (in MiB) required on a GPU to launch an agent (default: 2000)')
    return parser.parse_args()

def get_free_memory(gpu_id):
    """
    Returns the free memory (in MiB) for the given GPU ID by parsing nvidia-smi output.
    """
    command = f"nvidia-smi --id={gpu_id} --query-gpu=memory.free --format=csv,noheader,nounits"
    try:
        output = subprocess.check_output(command, shell=True)
        free_memory = int(output.decode().strip())
        return free_memory
    except Exception as e:
        print(f"Error getting free memory for GPU {gpu_id}: {e}")
        return 0

def cancel_sweep(sweep_id):
    """Cancel the sweep via WandB CLI."""
    cancel_cmd = f"wandb sweep --cancel {sweep_id}"
    try:
        subprocess.call(cancel_cmd, shell=True)
        print(f"Triggered sweep cancellation: {cancel_cmd}")
    except Exception as e:
        print(f"Failed to cancel sweep: {e}")

def main():
    args = parse_arguments()
    processes = []
    launched = 0
    sweep_id = args.sweep_id

    try:
        while launched < args.n_processes:
            assigned = False
            num_gpus = len(args.gpu_ids)
            # Start checking from an index that rotates with each new process.
            start_idx = launched % num_gpus
            idx = start_idx
            while not assigned:
                gpu_id = args.gpu_ids[idx]
                free_mem = get_free_memory(gpu_id)
                if free_mem >= args.min_free_vram:
                    found_gpu = gpu_id
                    assigned = True
                else:
                    print(f"GPU {gpu_id} has only {free_mem} MiB free (threshold: {args.min_free_vram} MiB).")
                    idx = (idx + 1) % num_gpus
                    if idx == start_idx:
                        print("No GPUs with sufficient free VRAM available on this cycle. Waiting 30 seconds before retrying...")
                        time.sleep(30)
                        idx = start_idx
            # Build the command: always "wandb agent <sweep_id>" with CUDA_VISIBLE_DEVICES set.
            command_to_run = f"CUDA_VISIBLE_DEVICES={found_gpu} wandb agent {sweep_id}"
            print(f"Launching agent {launched+1}/{args.n_processes} on GPU {found_gpu}: {command_to_run}")
            proc = subprocess.Popen(command_to_run, shell=True)
            processes.append(proc)
            launched += 1
            time.sleep(1)  # Stagger launches slightly

        # Wait for all launched processes to finish.
        for proc in processes:
            proc.wait()

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Terminating launched agents and cancelling sweep...")
        for proc in processes:
            proc.terminate()
        cancel_sweep(sweep_id)
        sys.exit(0)

if __name__ == '__main__':
    main()
