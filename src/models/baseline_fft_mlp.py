import torch
from torch import nn


class BaselineFFTMLP(nn.Module):
    """
    Supervised MLP baseline for FFT band-power MEG features.

    Input:
      - [B, D] or [D]
    Output:
      - [B, num_classes] or [num_classes]
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 2 or x.size(-1) != self.input_dim:
            raise ValueError(f"Expected [B, {self.input_dim}] or [{self.input_dim}], got {tuple(x.shape)}.")

        logits = self.classifier(x)

        if squeeze_batch:
            logits = logits.squeeze(0)
        return logits
