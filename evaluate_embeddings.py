from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data import DataSetType, TASK_TO_LABEL, VQVAE_DataSet
from src.model_factory import build_model


LABEL_TO_TASK = {label: task for task, label in TASK_TO_LABEL.items()}


def val_loss_from_name(path: Path) -> float:
    marker = "val_loss="
    if marker not in path.stem:
        return float("inf")
    return float(path.stem.rsplit(marker, maxsplit=1)[-1])


def list_checkpoints(checkpoint_dir: str) -> list[Path]:
    ckpt_dir = Path(checkpoint_dir)
    checkpoint_files = sorted(ckpt_dir.glob("epoch-*.ckpt"), key=val_loss_from_name)
    if checkpoint_files:
        return checkpoint_files

    last_checkpoint = ckpt_dir / "last.ckpt"
    if last_checkpoint.exists():
        return [last_checkpoint]

    raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir.resolve()}")


def find_best_checkpoint(checkpoint_dir: str) -> Path:
    return list_checkpoints(checkpoint_dir)[0]


def load_model_from_checkpoint(checkpoint_path: Path, config: Config, device: torch.device) -> torch.nn.Module:
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
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
    device: torch.device,
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
            x = x.to(device)
            embeddings = model.encode(x)
            pooled_embeddings = embeddings.mean(dim=1)
            features.append(pooled_embeddings.cpu().numpy())
            labels.append(y.cpu().numpy())

    return np.concatenate(features), np.concatenate(labels)


def evaluate_checkpoint(
    checkpoint_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> dict[str, float | str]:
    classifier = LogisticRegression(max_iter=2_000, class_weight="balanced")
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    labels = sorted(LABEL_TO_TASK)
    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred, labels=labels)

    row: dict[str, float | str] = {
        "checkpoint": str(checkpoint_path),
        "val_loss_from_filename": val_loss_from_name(checkpoint_path),
        "test_accuracy": accuracy,
    }
    for index, label in enumerate(labels):
        support = matrix[index].sum()
        row[f"{LABEL_TO_TASK[label]}_accuracy"] = matrix[index, index] / support if support else 0.0
    return row


def print_single_checkpoint_report(
    checkpoint_path: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    classifier = LogisticRegression(max_iter=2_000, class_weight="balanced")
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)

    labels = sorted(LABEL_TO_TASK)
    target_names = [LABEL_TO_TASK[label] for label in labels]
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Test accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=labels))


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
    parser.add_argument("--model-type", choices=["cnn", "lstm", "tcn", "transformer", "cnn2d"], default="cnn")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="Evaluate every epoch checkpoint and print a comparison table.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional CSV path for checkpoint comparison results.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device for encoder embedding extraction.",
    )
    args = parser.parse_args()

    config = Config(checkpoint_dir=args.checkpoint_dir, model_type=args.model_type)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    train_dataset = build_dataset("train", config)
    test_dataset = build_dataset("test", config)

    checkpoint_paths = list_checkpoints(config.checkpoint_dir) if args.all_checkpoints else [args.checkpoint or find_best_checkpoint(config.checkpoint_dir)]
    rows = []

    for checkpoint_path in checkpoint_paths:
        print(f"\nUsing checkpoint: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, config, device)
        x_train, y_train = extract_embeddings(model, train_dataset, args.batch_size, args.num_workers, device)
        x_test, y_test = extract_embeddings(model, test_dataset, args.batch_size, args.num_workers, device)

        if args.all_checkpoints:
            row = evaluate_checkpoint(checkpoint_path, x_train, y_train, x_test, y_test)
            rows.append(row)
            print(f"Test accuracy: {row['test_accuracy']:.4f}")
        else:
            print_single_checkpoint_report(checkpoint_path, x_train, y_train, x_test, y_test)

    if args.all_checkpoints:
        rows = sorted(rows, key=lambda row: float(row["test_accuracy"]), reverse=True)
        fieldnames = list(rows[0].keys()) if rows else []
        print("\nCheckpoint comparison:")
        for row in rows:
            print(
                f"{row['test_accuracy']:.4f} | val_loss={row['val_loss_from_filename']:.4f} | {row['checkpoint']}"
            )

        if args.output_csv is not None:
            args.output_csv.parent.mkdir(parents=True, exist_ok=True)
            with args.output_csv.open("w", newline="") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nWrote checkpoint comparison to {args.output_csv}")


if __name__ == "__main__":
    main()
