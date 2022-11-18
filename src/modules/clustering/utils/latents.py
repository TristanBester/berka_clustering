import torch


def calculate_latents(encoder, embedding, loader, device):
    latents = []

    for x, _ in loader:
        x = x.to(device)
        h = encoder(x)
        z, _, _ = embedding(h, device)
        latents.append(z)
    return torch.concatenate(latents)
