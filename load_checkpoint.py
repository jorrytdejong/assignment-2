from __future__ import annotations

from pathlib import Path
import sys

import torch

from src.auto_encoder import AutoEncoder
from src.config import Config
from src.model_factory import build_model


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
