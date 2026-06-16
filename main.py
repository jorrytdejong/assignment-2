# from src.vq_vae import VQVAE
from src.train import train_classifier, train_vae

from src.config import Config
from src.model_factory import build_autoencoder, build_classifier


def main():
    # model = VQVAE(input_size=248, num_layers=3, hidden_dim=64, num_pretrain_steps=20)
    config = Config(
        
    )

    print('Config:', config)

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
