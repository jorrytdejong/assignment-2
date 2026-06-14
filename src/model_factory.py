from src.auto_encoder import AutoEncoder
from src.config import Config
from src.models.cnn import ResNet1DDecoder, ResNet1DEncoder
from src.models.lstm import LSTMDecoder, LSTMEncoder


def build_model(config: Config) -> AutoEncoder:
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
        case _:
            raise ValueError(f"Unsupported model_type: {config.model_type}")

    return AutoEncoder(encoder=encoder, decoder=decoder, config=config)
