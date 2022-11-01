import torch.optim as optim

from ..models import ClusterModel
from ..modules.clustering.layers import DTCClusterLayer
from ..modules.clustering.loss import dtc_loss_handler
from ..modules.clustering.utils import calculate_centroids, calculate_latents


def _init_cluster_model(encoder, embedding, decoder, metric, n_clusters, loader):
    latents = calculate_latents(encoder, embedding, loader)
    centroids = calculate_centroids(latents, metric, n_clusters)
    dtc = DTCClusterLayer(encoder, embedding, centroids, metric)
    cl_model = ClusterModel(dtc, decoder, dtc_loss_handler)
    return cl_model


def train_clustering_layer(encoder, embedding, decoder, metric, n_clusters, loader):
    cl_model = _init_cluster_model(
        encoder, embedding, decoder, metric, n_clusters, loader
    )

    optimizer = optim.Adam(cl_model.parameters(), lr=0.0001)
    for i in range(10):
        for x, y in loader:
            loss = cl_model(x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print(loss.item())
    return cl_model
