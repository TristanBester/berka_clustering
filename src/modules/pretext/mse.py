import torch.nn.functional as F


def mse_loss_handler(encoder, embedding, decoder, x, device):
    h = encoder(x)
    z, _, _ = embedding(h, device)
    x_prime = decoder(z)
    return F.mse_loss(x_prime, x)
