from __future__ import annotations

from pathlib import Path
import sys

import torch

from src.auto_encoder import AutoEncoder
from src.config import Config
from src.models.cnn import ResNet1DDecoder, ResNet1DEncoder


def build_model(config: Config) -> AutoEncoder:
    encoder = ResNet1DEncoder(
        input_features=config.num_features,
        base_channels=config.base_channels,
        latent_dim=config.latent_dim,
    )
    decoder = ResNet1DDecoder(
        output_features=config.num_features,
        base_channels=config.base_channels,
        latent_dim=config.latent_dim,
    )
    return AutoEncoder(encoder=encoder, decoder=decoder, config=config)


def load_latest_checkpoint(checkpoint_dir: str = "checkpoints") -> AutoEncoder:
    config = Config(checkpoint_dir=checkpoint_dir)
    model = build_model(config)

    ckpt_dir = Path(checkpoint_dir)
    checkpoint_files = sorted(ckpt_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir.resolve()}")

    latest_checkpoint = checkpoint_files[-1]
    checkpoint = torch.load(latest_checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Loaded checkpoint: {latest_checkpoint}")
    return model


if __name__ == "__main__":
    checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints"
    load_latest_checkpoint(checkpoint_dir)
