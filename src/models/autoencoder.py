import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, encoder, embedding, decoder, pretext_loss_fn) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.decoder = decoder
        self.pretext_loss_fn = pretext_loss_fn

    def forward(self, x):
        return self.pretext_loss_fn(self.encoder, self.embedding, self.decoder, x)

    def embed(self, x):
        h = self.encoder(x)
        z, _, _ = self.embedding(h)
        return z
