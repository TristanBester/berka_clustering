import torch.optim as optim
import torch
from ..models import ClusterModel
from ..modules.clustering.layers import DTCClusterLayer
from ..modules.clustering.loss import dtc_loss_handler
from ..modules.clustering.utils import calculate_centroids, calculate_latents


def _init_cluster_model(encoder, embedding, decoder, metric_np, metric_torch, n_clusters, loader, device):
    with torch.no_grad():
        print('Calculating latents')
        latents = calculate_latents(encoder, embedding, loader, device)
        print('Calculating centroids')
        centroids = calculate_centroids(latents, metric_np, n_clusters)
        dtc = DTCClusterLayer(encoder, embedding, centroids, metric_torch, device)
        cl_model = ClusterModel(dtc, decoder, dtc_loss_handler)
        return cl_model


def train_clustering_layer(encoder, embedding, decoder, metric_np, metric_torch, n_clusters, loader, device):
    print('Initialising cluster model')
    cl_model = _init_cluster_model(
        encoder, embedding, decoder, metric_np, metric_torch, n_clusters, loader, device
    )
    optimizer = optim.Adam(cl_model.parameters(), lr=0.00001)
    batch_count = 0

    cl_model = cl_model.to(device)

    while batch_count < 100:
        for x, _ in loader:
            x = x.to(device)
            loss = cl_model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count += 1

            if batch_count > 100:
                break
            # print('CL', batch_count, loss.item())

    return cl_model
