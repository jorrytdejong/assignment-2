import torch
from torch import nn


class BaselineCNN1D(nn.Module):
    """
    Supervised 1D CNN baseline for MEG task classification.

    Input:
      - [B, T, F] or [T, F]
    Output:
      - [B, num_classes] or [num_classes]
    """

    def __init__(
        self,
        input_features: int = 248,
        num_classes: int = 4,
        dropout: float = 0.3,
        channels: tuple[int, ...] = (32, 64, 128),
        kernel_sizes: tuple[int, ...] = (7, 5, 3),
    ):
        super().__init__()
        if len(channels) != len(kernel_sizes):
            raise ValueError(
                f"Expected channels and kernel_sizes to have the same length, "
                f"got {len(channels)} and {len(kernel_sizes)}."
            )
        if not channels:
            raise ValueError("Expected at least one convolutional channel.")

        self.input_features = input_features

        layers: list[nn.Module] = []
        in_channels = input_features
        for out_channels, kernel_size in zip(channels, kernel_sizes, strict=True):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes),
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

        x = x.permute(0, 2, 1)
        logits = self.classifier(self.features(x))

        if squeeze_batch:
            logits = logits.squeeze(0)
        return logits
