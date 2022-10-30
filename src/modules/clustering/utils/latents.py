import torch


def calculate_latents(encoder, embedding, loader):
    latents = []

    for x, _ in loader:
        h = encoder(x)
        z, _, _ = embedding(h)
        latents.append(z)
    return torch.concatenate(latents)
