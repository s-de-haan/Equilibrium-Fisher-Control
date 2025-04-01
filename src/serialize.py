import json
import torch
import os
import re
import shutil

_DUMP_DIR = "./dumps"

def clear_dumps():
    shutil.rmtree(_DUMP_DIR)

def dump_tensor(tensor: torch.Tensor, name: str):
    tensor_list = tensor.tolist()
    path = f"{_DUMP_DIR}/{name}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(tensor_list, f)

def next_available_directory(name: str) -> str:
    directory_count = -1  # Start at -1 so 0 is the first available
    name_regex = re.compile(rf"{name}_(\d+)")
    for file in os.listdir(_DUMP_DIR):
        match = name_regex.fullmatch(file)
        if match:
            directory_count = max(directory_count, int(match.group(1)))
    return f"{name}_{directory_count + 1}"


def load_tensor(name: str) -> torch.Tensor:
    path = f"./dumps/{name}.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File {path} does not exist")
    with open(path, "r") as f:
        tensor_list = json.load(f)
    return torch.tensor(tensor_list)
