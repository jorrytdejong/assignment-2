import torch
from torch import nn


class BaselineLSTM(nn.Module):
    """
    Supervised LSTM baseline for MEG task classification.

    Input:
      - [B, T, F] or [T, F]
    Output:
      - [B, num_classes] or [num_classes]
    """

    def __init__(
        self,
        input_features: int = 248,
        hidden_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=lstm_dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )

        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(output_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeeze_batch = False
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze_batch = True

        if x.dim() != 3 or x.size(-1) != self.input_features:
            raise ValueError(
                f"Expected [B, T, {self.input_features}] or [T, {self.input_features}], got {tuple(x.shape)}."
            )

        outputs, _ = self.lstm(x)
        pooled = outputs.mean(dim=1)
        logits = self.classifier(pooled)

        if squeeze_batch:
            logits = logits.squeeze(0)
        return logits
