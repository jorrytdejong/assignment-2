from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from load_checkpoint import build_model
from src.config import Config
from src.data import DataSetType, TASK_TO_LABEL, VQVAE_DataSet


LABEL_TO_TASK = {label: task for task, label in TASK_TO_LABEL.items()}


def find_best_checkpoint(checkpoint_dir: str) -> Path:
    ckpt_dir = Path(checkpoint_dir)
    checkpoint_files = sorted(ckpt_dir.glob("epoch-*.ckpt"))
    if not checkpoint_files:
        last_checkpoint = ckpt_dir / "last.ckpt"
        if last_checkpoint.exists():
            return last_checkpoint
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir.resolve()}")

    def val_loss_from_name(path: Path) -> float:
        marker = "val_loss="
        if marker not in path.stem:
            return float("inf")
        return float(path.stem.rsplit(marker, maxsplit=1)[-1])

    return min(checkpoint_files, key=val_loss_from_name)


def load_model_from_checkpoint(checkpoint_path: Path, config: Config) -> torch.nn.Module:
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def build_dataset(split: str, config: Config) -> VQVAE_DataSet:
    dataset = VQVAE_DataSet(
        DataSetType.INTRA,
        split,
        window_size=config.window_size,
        window_stride=config.window_stride,
    )
    dataset.load()
    return dataset


def extract_embeddings(
    model: torch.nn.Module,
    dataset: VQVAE_DataSet,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    features = []
    labels = []
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc=f"extracting {dataset.split} embeddings"):
            embeddings = model.encode(x)
            pooled_embeddings = embeddings.mean(dim=1)
            features.append(pooled_embeddings.cpu().numpy())
            labels.append(y.cpu().numpy())

    return np.concatenate(features), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate task-label accuracy from frozen autoencoder embeddings."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint to evaluate. Defaults to the lowest val_loss checkpoint in checkpoint_dir.",
    )
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    config = Config(checkpoint_dir=args.checkpoint_dir)
    checkpoint_path = args.checkpoint or find_best_checkpoint(config.checkpoint_dir)
    print(f"Using checkpoint: {checkpoint_path}")

    model = load_model_from_checkpoint(checkpoint_path, config)
    train_dataset = build_dataset("train", config)
    test_dataset = build_dataset("test", config)

    x_train, y_train = extract_embeddings(model, train_dataset, args.batch_size, args.num_workers)
    x_test, y_test = extract_embeddings(model, test_dataset, args.batch_size, args.num_workers)

    classifier = LogisticRegression(max_iter=2_000, class_weight="balanced")
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    labels = sorted(LABEL_TO_TASK)
    target_names = [LABEL_TO_TASK[label] for label in labels]
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))


if __name__ == "__main__":
    main()
