import torch
from torch.utils.data import Dataset, DataLoader

class ContinualLearningDataset(Dataset):
    def __init__(self, task_id, n_samples=50, noise=0.1):
        self.data = torch.zeros(n_samples, 4, 2)
        self.task_id = task_id
        self.labels = torch.zeros(n_samples)
        
        if task_id == 1:  # XOR on diagonals
            for i in range(n_samples):
                if i < n_samples/2:  # Label 1
                    points = torch.tensor([
                        [1.0, 0.0], 
                        [0.0, 1.0],
                        [0.5, 0.5], 
                        [0.25, 0.75]
                    ])
                else:  # Label 0
                    points = torch.tensor([
                        [0.0, 0.0], 
                        [1.0, 1.0],
                        [0.5, 0.5], 
                        [0.75, 0.25]
                    ])
                self.data[i] = points + torch.randn_like(points) * noise
                self.labels[i] = 1 if i < n_samples/2 else 0
                
        elif task_id == 2:  # Square vs Diamond
            for i in range(n_samples):
                if i < n_samples/2:  # Label 1: Diamond
                    points = torch.tensor([
                        [0.0, 0.5],
                        [0.5, 0.0],
                        [0.5, 1.0],
                        [1.0, 0.5]
                    ])
                else:  # Label 0: Square
                    points = torch.tensor([
                        [0.0, 0.0],
                        [1.0, 0.0],
                        [1.0, 1.0],
                        [0.0, 1.0]
                    ])
                self.data[i] = points + torch.randn_like(points) * noise
                self.labels[i] = 1 if i < n_samples/2 else 0
                
        elif task_id == 3:  # Close pairs vs Distant pairs
            for i in range(n_samples):
                if i < n_samples/2:  # Label 1: Close pairs
                    points = torch.tensor([
                        [0.3, 0.3],
                        [0.4, 0.4],
                        [0.6, 0.6],
                        [0.7, 0.7]
                    ])
                else:  # Label 0: Distant pairs
                    points = torch.tensor([
                        [0.1, 0.1],
                        [0.9, 0.9],
                        [0.1, 0.9],
                        [0.9, 0.1]
                    ])
                self.data[i] = points + torch.randn_like(points) * noise
                self.labels[i] = 1 if i < n_samples/2 else 0
        
        def to_one_hot(x):
            n = len(x)
            result = torch.zeros((n, 2))
            result[range(n), x] = 1
            return result

        self.labels = to_one_hot(self.labels.int())

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.task_id

def get_dataloader(task_id, batch_size=32):
    dataset = ContinualLearningDataset(task_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)