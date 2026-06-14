import torch
from torch import nn


class Conv2DBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class CNN2DEncoder(nn.Module):
    def __init__(
        self,
        input_features: int = 248,
        base_channels: int = 32,
        latent_dim: int = 64,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.input_features = input_features
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[Conv2DBlock(base_channels) for _ in range(num_blocks)])
        self.head = nn.Conv1d(base_channels, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 3 or x.size(-1) != self.input_features:
            raise ValueError(
                f"Expected [B, T, {self.input_features}] or [T, {self.input_features}], got {tuple(x.shape)}."
            )

        x = x.unsqueeze(1)
        features = self.blocks(self.stem(x))
        features = features.mean(dim=-1)
        z = self.head(features).permute(0, 2, 1)

        if squeeze_batch:
            z = z.squeeze(0)
        return z


class CNN2DDecoder(nn.Module):
    def __init__(
        self,
        output_features: int = 248,
        base_channels: int = 32,
        latent_dim: int = 64,
        num_blocks: int = 3,
    ):
        super().__init__()
        self.output_features = output_features
        self.input_projection = nn.Conv1d(latent_dim, base_channels, kernel_size=1)
        self.blocks = nn.Sequential(*[Conv2DBlock(base_channels) for _ in range(num_blocks)])
        self.head = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze_batch = True

        if z.dim() != 3:
            raise ValueError(f"Expected [B, T, C] or [T, C], got {tuple(z.shape)}.")

        z = z.permute(0, 2, 1)
        features = self.input_projection(z)
        features = features.unsqueeze(-1).expand(-1, -1, -1, self.output_features)
        x_hat = self.head(self.blocks(features)).squeeze(1)

        if squeeze_batch:
            x_hat = x_hat.squeeze(0)
        return x_hat
