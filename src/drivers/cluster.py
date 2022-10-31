import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from umap import UMAP


def _calculate_latents(model, loader):
    latents = []
    for x, _ in loader:
        latents.append(model.embed(x))
    return torch.concatenate(latents)


def calculate_clusters_from_latents(
    model, loader, k=3, dim_reduction=None, lower_dim=5
):
    latents = _calculate_latents(model, loader)
    latents = latents.squeeze(1).detach().numpy()

    if dim_reduction == "PCA":
        latents = PCA(n_components=lower_dim).fit_transform(latents)
    elif dim_reduction == "UMAP":
        latents = UMAP(n_components=lower_dim).fit_transform(latents)

    clusters = KMeans(n_clusters=k).fit_predict(latents)
    return clusters


def calculate_clusters_from_layer(model, loader):
    assignments = []

    for x, _ in loader:
        _, Q, _ = model.cluster_layer(x)
        assignments.append(Q)
    assignments = torch.concatenate(assignments)
    return assignments.argmax(dim=-1)
