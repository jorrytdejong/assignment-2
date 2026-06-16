from src.auto_encoder import AutoEncoder
from src.classifier import Classifier
from src.config import Config
from src.models.baseline_cnn1d import BaselineCNN1D
from src.models.cnn2d import CNN2DDecoder, CNN2DEncoder
from src.models.cnn import ResNet1DDecoder, ResNet1DEncoder
from src.models.lstm import LSTMDecoder, LSTMEncoder
from src.models.tcn import TCNDecoder, TCNEncoder
from src.models.transformer import TransformerDecoderModel, TransformerEncoderModel


def build_autoencoder(config: Config) -> AutoEncoder:
    match config.model_type:
        case "cnn":
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
        case "lstm":
            encoder = LSTMEncoder(
                input_features=config.num_features,
                hidden_dim=config.lstm_hidden_dim,
                latent_dim=config.latent_dim,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout,
            )
            decoder = LSTMDecoder(
                output_features=config.num_features,
                hidden_dim=config.lstm_hidden_dim,
                latent_dim=config.latent_dim,
                num_layers=config.lstm_num_layers,
                dropout=config.lstm_dropout,
            )
        case "tcn":
            encoder = TCNEncoder(
                input_features=config.num_features,
                channels=config.tcn_channels,
                latent_dim=config.latent_dim,
                num_blocks=config.tcn_num_blocks,
                kernel_size=config.tcn_kernel_size,
                dropout=config.tcn_dropout,
            )
            decoder = TCNDecoder(
                output_features=config.num_features,
                channels=config.tcn_channels,
                latent_dim=config.latent_dim,
                num_blocks=config.tcn_num_blocks,
                kernel_size=config.tcn_kernel_size,
                dropout=config.tcn_dropout,
            )
        case "transformer":
            encoder = TransformerEncoderModel(
                input_features=config.num_features,
                d_model=config.transformer_d_model,
                latent_dim=config.latent_dim,
                num_heads=config.transformer_num_heads,
                num_layers=config.transformer_num_layers,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout,
            )
            decoder = TransformerDecoderModel(
                output_features=config.num_features,
                d_model=config.transformer_d_model,
                latent_dim=config.latent_dim,
                num_heads=config.transformer_num_heads,
                num_layers=config.transformer_num_layers,
                dim_feedforward=config.transformer_dim_feedforward,
                dropout=config.transformer_dropout,
            )
        case "cnn2d":
            encoder = CNN2DEncoder(
                input_features=config.num_features,
                base_channels=config.cnn2d_base_channels,
                latent_dim=config.latent_dim,
                num_blocks=config.cnn2d_num_blocks,
            )
            decoder = CNN2DDecoder(
                output_features=config.num_features,
                base_channels=config.cnn2d_base_channels,
                latent_dim=config.latent_dim,
                num_blocks=config.cnn2d_num_blocks,
            )
        case _:
            raise ValueError(f"Unsupported autoencoder model_type: {config.model_type}")

    return AutoEncoder(encoder=encoder, decoder=decoder, config=config)


def build_classifier(config: Config) -> Classifier:
    match config.model_type:
        case "baseline_cnn1d":
            model = BaselineCNN1D(
                input_features=config.num_features,
                num_classes=config.num_classes,
                dropout=config.baseline_cnn_dropout,
                channels=config.baseline_cnn_channels,
                kernel_sizes=config.baseline_cnn_kernel_sizes,
            )
        case _:
            raise ValueError(f"Unsupported classifier model_type: {config.model_type}")

    return Classifier(model=model, config=config)


def build_model(config: Config) -> AutoEncoder:
    return build_autoencoder(config)
