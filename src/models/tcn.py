import torch
from torch import nn


class TemporalResidualBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class TCNEncoder(nn.Module):
    def __init__(
        self,
        input_features: int = 248,
        channels: int = 128,
        latent_dim: int = 64,
        num_blocks: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_features = input_features
        self.stem = nn.Sequential(
            nn.Conv1d(input_features, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**block_index,
                    dropout=dropout,
                )
                for block_index in range(num_blocks)
            ]
        )
        self.head = nn.Conv1d(channels, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 3 or x.size(-1) != self.input_features:
            raise ValueError(
                f"Expected [B, T, {self.input_features}] or [T, {self.input_features}], got {tuple(x.shape)}."
            )

        x = x.permute(0, 2, 1)
        z = self.head(self.blocks(self.stem(x)))
        z = z.permute(0, 2, 1)

        if squeeze_batch:
            z = z.squeeze(0)
        return z


class TCNDecoder(nn.Module):
    def __init__(
        self,
        output_features: int = 248,
        channels: int = 128,
        latent_dim: int = 64,
        num_blocks: int = 5,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(latent_dim, channels, kernel_size=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *[
                TemporalResidualBlock(
                    channels=channels,
                    kernel_size=kernel_size,
                    dilation=2**block_index,
                    dropout=dropout,
                )
                for block_index in range(num_blocks)
            ]
        )
        self.head = nn.Conv1d(channels, output_features, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze_batch = True

        if z.dim() != 3:
            raise ValueError(f"Expected [B, T, C] or [T, C], got {tuple(z.shape)}.")

        z = z.permute(0, 2, 1)
        x_hat = self.head(self.blocks(self.stem(z)))
        x_hat = x_hat.permute(0, 2, 1)

        if squeeze_batch:
            x_hat = x_hat.squeeze(0)
        return x_hat
