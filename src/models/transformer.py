import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10_000):
        super().__init__()
        positions = torch.arange(max_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_terms)
        pe[:, 1::2] = torch.cos(positions * div_terms[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds positional encoding limit {self.pe.size(1)}.")
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderModel(nn.Module):
    def __init__(
        self,
        input_features: int = 248,
        d_model: int = 128,
        latent_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_features = input_features
        self.input_projection = nn.Linear(input_features, d_model)
        self.position = SinusoidalPositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 3 or x.size(-1) != self.input_features:
            raise ValueError(
                f"Expected [B, T, {self.input_features}] or [T, {self.input_features}], got {tuple(x.shape)}."
            )

        hidden = self.position(self.input_projection(x))
        z = self.output_projection(self.transformer(hidden))

        if squeeze_batch:
            z = z.squeeze(0)
        return z


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        output_features: int = 248,
        d_model: int = 128,
        latent_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(latent_dim, d_model)
        self.position = SinusoidalPositionalEncoding(d_model)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if z.dim() == 2:
            z = z.unsqueeze(0)
            squeeze_batch = True

        if z.dim() != 3:
            raise ValueError(f"Expected [B, T, C] or [T, C], got {tuple(z.shape)}.")

        hidden = self.position(self.input_projection(z))
        x_hat = self.output_projection(self.transformer(hidden))

        if squeeze_batch:
            x_hat = x_hat.squeeze(0)
        return x_hat
