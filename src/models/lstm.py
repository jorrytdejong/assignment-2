import torch
from torch import nn


class LSTMEncoder(nn.Module):
    """
    LSTM encoder for inputs shaped [B, T, input_features].

    Returns a latent sequence shaped [B, T, latent_dim] so it can be used by the
    existing autoencoder and contrastive-loss code.
    """

    def __init__(
        self,
        input_features: int = 248,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_features = input_features
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 3 or x.size(-1) != self.input_features:
            raise ValueError(
                f"Expected [B, T, {self.input_features}] or [T, {self.input_features}], got {tuple(x.shape)}."
            )

        hidden_states, _ = self.lstm(x)
        z = self.projection(hidden_states)

        if squeeze_batch:
            z = z.squeeze(0)
        return z


class LSTMDecoder(nn.Module):
    """
    LSTM decoder that maps latent sequences [B, T, latent_dim] back to features.
    """

    def __init__(
        self,
        output_features: int = 248,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.projection = nn.Linear(hidden_dim, output_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze_batch = True

        if z.dim() != 3:
            raise ValueError(f"Expected [B, T, C] or [T, C], got {tuple(z.shape)}.")

        hidden_states, _ = self.lstm(z)
        x_hat = self.projection(hidden_states)

        if squeeze_batch:
            x_hat = x_hat.squeeze(0)
        return x_hat
