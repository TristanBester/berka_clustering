import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


def _calculate_similarity_matrix(X, metric):
    similarity_matrix = torch.zeros(size=(X.shape[0], X.shape[0]))

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            similarity_matrix[i][j] = metric(X[i], X[j])
    return similarity_matrix


def calculate_centroids(latents, metric, k):
    similarity_matrix = _calculate_similarity_matrix(latents, metric)
    similarity_matrix = similarity_matrix.detach().numpy().astype(float)
    similarity_matrix = np.nan_to_num(similarity_matrix)

    clustering_assignments = AgglomerativeClustering(
        n_clusters=k,
        affinity="precomputed",
        linkage="complete",
    ).fit_predict(similarity_matrix)

    centroids = []
    for i in np.unique(clustering_assignments):
        centroid = (
            latents[clustering_assignments == i].mean(dim=0, dtype=float).unsqueeze(0)
        )
        centroids.append(centroid)
    centroids = torch.cat(centroids)
    return centroids
