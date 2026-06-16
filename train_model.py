from __future__ import annotations

import argparse

from src.config import Config
from src.model_factory import build_autoencoder, build_classifier
from src.train import train_classifier, train_vae


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an autoencoder or supervised baseline model.")
    parser.add_argument("--task-type", choices=["autoencoder", "classifier"], default="autoencoder")
    parser.add_argument("--model-type", choices=["cnn", "lstm", "tcn", "transformer", "cnn2d", "baseline_cnn1d"], default="cnn")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--window-size", type=int, default=None)
    parser.add_argument("--window-stride", type=int, default=None)
    parser.add_argument("--downsample-factor", type=int, default=None)
    parser.add_argument("--preprocess-mode", choices=["stride", "block_mean"], default=None)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "mps", "auto"], default=None)
    parser.add_argument("--precision", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--num-classes", type=int, default=None)
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
    parser.add_argument("--baseline-cnn-dropout", type=float, default=None)
    args = parser.parse_args()

    config = Config(task_type=args.task_type, model_type=args.model_type)
    for field_name, value in vars(args).items():
        if value is not None and hasattr(config, field_name):
            setattr(config, field_name, value)

    print("Config:", config)
    if config.task_type == "autoencoder":
        model = build_autoencoder(config)
        train_vae(model, config)
    elif config.task_type == "classifier":
        model = build_classifier(config)
        train_classifier(model, config)
    else:
        raise ValueError(f"Unsupported task_type: {config.task_type}")


if __name__ == "__main__":
    main()
