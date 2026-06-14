# from src.vq_vae import VQVAE
from src.train import train_vae

from src.config import Config
from src.model_factory import build_model


def main():
    # model = VQVAE(input_size=248, num_layers=3, hidden_dim=64, num_pretrain_steps=20)
    config = Config(
        
    )

    print('Config:', config)
    model = build_model(config)

    train_vae(model, config)

if __name__ == "__main__":
    main()
