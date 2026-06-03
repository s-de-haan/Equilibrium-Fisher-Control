import torch
import torch.nn as nn


def normal_init(module: nn.Module, mean: float, std: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=mean, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def labels_to_onehot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.to(dtype=torch.long)
    onehot = torch.zeros(labels.size(0), num_classes, device=labels.device)
    onehot.scatter_(1, labels.view(-1, 1), 1.0)
    return onehot


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        hidden_dim: int = 200,
        image_dim: int = 784,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.image_dim = image_dim
        self.fc1_1 = nn.Linear(latent_dim, hidden_dim)
        self.fc1_1_bn = nn.BatchNorm1d(hidden_dim)

        self.fc1_2 = nn.Linear(num_classes, hidden_dim)
        self.fc1_2_bn = nn.BatchNorm1d(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, image_dim)

    def weight_init(self, mean: float, std: float) -> None:
        for module in self.modules():
            normal_init(module, mean, std)

    def forward(self, z: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1_1_bn(self.fc1_1(z)))
        y = torch.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], dim=1)
        x = torch.relu(self.fc2_bn(self.fc2(x)))
        x = torch.relu(self.fc3_bn(self.fc3(x)))
        return torch.tanh(self.fc4(x))

    @torch.no_grad()
    def generate_from_labels(
        self,
        labels: torch.Tensor,
        z: torch.Tensor | None = None,
        *,
        reshape: bool = True,
        image_size: tuple[int, int] = (28, 28),
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        labels = labels.view(-1).to(device=self.fc1_1.weight.device, dtype=torch.long)

        if z is None:
            z = torch.rand(
                labels.size(0),
                self.latent_dim,
                device=labels.device,
                generator=generator,
            )
        else:
            z = z.to(device=labels.device, dtype=self.fc1_1.weight.dtype)

        labels_onehot = labels_to_onehot(labels, self.num_classes).to(dtype=z.dtype)
        images = self(z, labels_onehot)

        if reshape:
            height, width = image_size
            if height * width != self.image_dim:
                raise ValueError(
                    f"image_size={image_size} does not match image_dim={self.image_dim}"
                )
            return images.view(-1, 1, height, width)

        return images


class Discriminator(nn.Module):
    def __init__(
        self,
        image_dim: int = 784,
        num_classes: int = 10,
        hidden_dim: int = 200,
    ) -> None:
        super().__init__()
        self.fc1_1 = nn.Linear(image_dim, hidden_dim)
        self.fc1_2 = nn.Linear(num_classes, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2_bn = nn.BatchNorm1d(hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_bn = nn.BatchNorm1d(hidden_dim)

        self.fc4 = nn.Linear(hidden_dim, 1)

    def weight_init(self, mean: float, std: float) -> None:
        for module in self.modules():
            normal_init(module, mean, std)

    def forward(self, img: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.leaky_relu(self.fc1_1(img), 0.2)
        y = torch.nn.functional.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], dim=1)
        x = torch.nn.functional.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = torch.nn.functional.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        return torch.sigmoid(self.fc4(x))
