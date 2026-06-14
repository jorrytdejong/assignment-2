from __future__ import annotations

import argparse

from src.config import Config
from src.model_factory import build_model
from src.train import train_vae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an autoencoder model.")
    parser.add_argument("--model-type", choices=["cnn", "lstm", "tcn", "transformer", "cnn2d"], default="cnn")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "mps", "auto"], default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--lstm-hidden-dim", type=int, default=None)
    parser.add_argument("--lstm-num-layers", type=int, default=None)
    parser.add_argument("--lstm-dropout", type=float, default=None)
    parser.add_argument("--tcn-channels", type=int, default=None)
    parser.add_argument("--tcn-num-blocks", type=int, default=None)
    parser.add_argument("--tcn-kernel-size", type=int, default=None)
    parser.add_argument("--tcn-dropout", type=float, default=None)
    parser.add_argument("--transformer-d-model", type=int, default=None)
    parser.add_argument("--transformer-num-heads", type=int, default=None)
    parser.add_argument("--transformer-num-layers", type=int, default=None)
    parser.add_argument("--transformer-dim-feedforward", type=int, default=None)
    parser.add_argument("--transformer-dropout", type=float, default=None)
    parser.add_argument("--cnn2d-base-channels", type=int, default=None)
    parser.add_argument("--cnn2d-num-blocks", type=int, default=None)
    args = parser.parse_args()

    config = Config(model_type=args.model_type)
    for field_name, value in vars(args).items():
        if value is not None and hasattr(config, field_name):
            setattr(config, field_name, value)

    print("Config:", config)
    model = build_model(config)
    train_vae(model, config)


if __name__ == "__main__":
    main()
