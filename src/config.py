import dataclasses
from typing import Literal


@dataclasses.dataclass
class Config:
    
    # training
    learning_rate: float = 1e-3
    batch_size: int = 1
    num_workers: int = 4
    
    max_epochs: int = 10
    val_check_interval: float = 1.0
    enable_checkpointing: bool = True
    enable_progress_bar: bool = True
    enable_model_summary: bool = True

    accelerator: str = 'cpu'
    log_every_n_steps: int = 10
    precision: str = "16-mixed"

    # callbacks
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 1e-3
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    stochastic_weight_averaging_swa_lrs: float = 1e-3
    stochastic_weight_averaging_swa_epoch_start: int = 10

    # contrastive loss
    weight_mse: float = 1.0
    weight_contrastive: float = 1.0

    # dataset windowing
    dataset_type: Literal["intra", "cross"] = "intra"
    window_size: int = 2048
    window_stride: int = 2048
    downsample_factor: int = 20
    preprocess_mode: Literal["stride", "block_mean"] = "stride"
    
    
    ### MODELS ###

    task_type: Literal["autoencoder", "classifier"] = "autoencoder"
    model_type: Literal["cnn", "lstm", "tcn", "transformer", "cnn2d", "baseline_cnn1d"] = "cnn"
    
    ## CNN ##
    
    num_features: int = 248
    base_channels: int = 128
    latent_dim: int = 64
    num_classes: int = 4

    ## LSTM ##

    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0

    ## TCN ##

    tcn_channels: int = 128
    tcn_num_blocks: int = 5
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1

    ## Transformer ##

    transformer_d_model: int = 128
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1

    ## 2D CNN ##

    cnn2d_base_channels: int = 32
    cnn2d_num_blocks: int = 3

    ## Baseline 1D CNN classifier ##

    baseline_cnn_channels: tuple[int, ...] = (32, 64, 128)
    baseline_cnn_kernel_sizes: tuple[int, ...] = (7, 5, 3)
    baseline_cnn_dropout: float = 0.3
