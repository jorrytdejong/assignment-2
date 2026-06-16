import lightning as L
import torch
from torch import nn

from src.config import Config


class Classifier(L.LightningModule):
    def __init__(self, model: nn.Module, config: Config):
        super().__init__()
        self.model = model
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        self.validation_file_indices: list[torch.Tensor] = []
        self.validation_predictions: list[torch.Tensor] = []
        self.validation_labels: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        if len(batch) == 3:
            x, y, file_indices = batch
        else:
            x, y = batch
            file_indices = None

        logits = self(x)
        loss = self.loss_fn(logits, y)
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean()

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        if stage == "val" and file_indices is not None:
            self.validation_file_indices.append(file_indices.detach().cpu())
            self.validation_predictions.append(predictions.detach().cpu())
            self.validation_labels.append(y.detach().cpu())

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def on_validation_epoch_start(self):
        self.validation_file_indices.clear()
        self.validation_predictions.clear()
        self.validation_labels.clear()

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self):
        if not self.validation_file_indices:
            return

        file_indices = torch.cat(self.validation_file_indices)
        predictions = torch.cat(self.validation_predictions)
        labels = torch.cat(self.validation_labels)

        correct_files = 0
        unique_file_indices = torch.unique(file_indices)
        for file_idx in unique_file_indices:
            mask = file_indices == file_idx
            file_predictions = predictions[mask]
            file_label = labels[mask][0]
            vote_counts = torch.bincount(file_predictions, minlength=self.config.num_classes)
            file_prediction = vote_counts.argmax()
            correct_files += int(file_prediction == file_label)

        file_accuracy = torch.tensor(
            correct_files / len(unique_file_indices),
            dtype=torch.float32,
            device=self.device,
        )
        self.log("val_file_acc", file_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
