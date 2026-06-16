import lightning as L
# from src.vq_vae import VQVAE
from src.auto_encoder import AutoEncoder
from src.classifier import Classifier
from src.data import DataSet, DataSetType
from src.data import MEGBaselineWindowDataset
from src.data import VQVAE_DataSet
from src.config import Config


from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary, RichProgressBar, StochasticWeightAveraging


def _dataset_type_from_config(config: Config) -> DataSetType:
    match config.dataset_type:
        case "intra":
            return DataSetType.INTRA
        case "cross":
            return DataSetType.CROSS
        case _:
            raise ValueError(f"Unsupported dataset_type: {config.dataset_type}")


def _classifier_val_splits(config: Config) -> tuple[str, ...]:
    if config.dataset_type == "cross":
        return ("test1", "test2", "test3")
    return ("test",)


def _autoencoder_val_splits(config: Config) -> tuple[str, ...]:
    if config.dataset_type == "cross":
        return ("test1",)
    return ("test",)


def train_vae(model: AutoEncoder, config: Config):
    dataset_type = _dataset_type_from_config(config)
    val_splits = _autoencoder_val_splits(config)

    train_dataset = VQVAE_DataSet(
        dataset_type,
        'train',
        window_size=config.window_size,
        window_stride=config.window_stride,
    )
    train_dataset.load()
    
    val_dataset = VQVAE_DataSet(
        dataset_type,
        val_splits,
        window_size=config.window_size,
        window_stride=config.window_stride,
    )
    val_dataset.load()
    val_dataset.get_mean_and_std(train_dataset)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers) # type: ignore
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers) # type: ignore

    callbacks = [
        ModelSummary(max_depth=3),
        EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, mode='min', min_delta=config.early_stopping_min_delta, check_on_train_epoch_end=False),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="epoch-{epoch:02d}-val_loss-{val_loss:.4f}",
            monitor='val_loss',
            mode='min',
            save_top_k=-1,
            every_n_epochs=1,
            save_last=True,
            save_weights_only=True,
        ),
        RichProgressBar(),
    ]
    if config.stochastic_weight_averaging_swa_epoch_start < config.max_epochs:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.stochastic_weight_averaging_swa_lrs,
                swa_epoch_start=config.stochastic_weight_averaging_swa_epoch_start,
            )
        )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        # logger=L.loggers.TensorBoardLogger("logs"),
        logger=TensorBoardLogger(save_dir=config.log_dir),
        enable_checkpointing=config.enable_checkpointing,
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        # enable_sanity_check=True,
        accelerator=config.accelerator,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=callbacks,
        precision=config.precision, # type: ignore
    )
    trainer.fit(model, train_dataloader, val_dataloader)


def train_classifier(model: Classifier, config: Config):
    dataset_type = _dataset_type_from_config(config)
    val_splits = _classifier_val_splits(config)

    train_dataset = MEGBaselineWindowDataset(
        dataset_type,
        'train',
        downsample_factor=config.downsample_factor,
        window_size=config.window_size,
        window_stride=config.window_stride,
        preprocess_mode=config.preprocess_mode,
    )
    train_dataset.load()

    val_dataset = MEGBaselineWindowDataset(
        dataset_type,
        val_splits,
        downsample_factor=config.downsample_factor,
        window_size=config.window_size,
        window_stride=config.window_stride,
        preprocess_mode=config.preprocess_mode,
        return_file_index=True,
    )
    val_dataset.load()

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers) # type: ignore
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers) # type: ignore

    callbacks = [
        ModelSummary(max_depth=3),
        EarlyStopping(monitor='val_file_acc', patience=config.early_stopping_patience, mode='max', min_delta=config.early_stopping_min_delta, check_on_train_epoch_end=False),
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="epoch-{epoch:02d}-val_file_acc-{val_file_acc:.4f}-val_acc-{val_acc:.4f}",
            monitor='val_file_acc',
            mode='max',
            save_top_k=-1,
            every_n_epochs=1,
            save_last=True,
            save_weights_only=True,
        ),
        RichProgressBar(),
    ]
    if config.stochastic_weight_averaging_swa_epoch_start < config.max_epochs:
        callbacks.append(
            StochasticWeightAveraging(
                swa_lrs=config.stochastic_weight_averaging_swa_lrs,
                swa_epoch_start=config.stochastic_weight_averaging_swa_epoch_start,
            )
        )

    trainer = L.Trainer(
        max_epochs=config.max_epochs,
        val_check_interval=config.val_check_interval,
        logger=TensorBoardLogger(save_dir=config.log_dir),
        enable_checkpointing=config.enable_checkpointing,
        enable_progress_bar=config.enable_progress_bar,
        enable_model_summary=config.enable_model_summary,
        accelerator=config.accelerator,
        log_every_n_steps=config.log_every_n_steps,
        callbacks=callbacks,
        precision=config.precision, # type: ignore
    )
    trainer.fit(model, train_dataloader, val_dataloader)
