import torch
import torch.nn.functional as F


def _vae_loss(x, decoded, z_mean, z_log_var):
    """ELBO loss"""
    # Sum over latent dimension
    kl_div = (
        -0.5
        * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=-1).mean()
    )
    mse_loss = F.mse_loss(decoded, x)
    return mse_loss + kl_div


def vae_loss_handler(encoder, embedding, decoder, x, device):
    h = encoder(x)
    z, z_mean, z_log_var = embedding(h, device)
    x_prime = decoder(z)
    return _vae_loss(x, x_prime, z_mean, z_log_var)
