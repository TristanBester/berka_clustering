import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.encoder = config["encoder"](**config["encoder_kwargs"])
        self.embedding = config["embedding"](**config["embedding_kwargs"])
        self.decoder = config["decoder"](**config["decoder_kwargs"])
        self.pretext_loss_fn = config["pretext"]

    def forward(self, x):
        return self.pretext_loss_fn(self.encoder, self.embedding, self.decoder, x)
