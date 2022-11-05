from multiprocessing import Pool

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering


def _calc(data):
    i, j, x_i, x_j, metric = data
    res = metric(x_i, x_j)
    return (i, j, res)

def _calculate_similarity_matrix(X, metric):
    X = X.cpu().detach().numpy()

    similarity_matrix = np.zeros((X.shape[0], X.shape[0]))
    work = []

    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            work.append((i, j, X[i], X[j], metric))
    
    print('Calc')
    with Pool(8) as p:
        results = p.map(_calc, work, chunksize=500)
    
    for res in results:
        similarity_matrix[res[0]][res[1]] = res[2].item()

    return similarity_matrix


def calculate_centroids(latents, metric, k):
    print('Calcuating similarity martrix')
    similarity_matrix = _calculate_similarity_matrix(latents.clone(), metric)
    similarity_matrix = similarity_matrix.astype(float)
    similarity_matrix = np.nan_to_num(similarity_matrix)

    print('Performing clustering')
    clustering_assignments = AgglomerativeClustering(
        n_clusters=k,
        affinity="precomputed",
        linkage="complete",
    ).fit_predict(similarity_matrix)

    print('Aggregating results')
    centroids = []
    for i in np.unique(clustering_assignments):
        centroid = (
            latents[clustering_assignments == i].mean(dim=0, dtype=float).unsqueeze(0)
        )
        centroids.append(centroid)
    centroids = torch.cat(centroids)
    return centroids
