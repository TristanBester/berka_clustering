import torch
import torch.nn as nn


class DTCClusterLayer(nn.Module):
    def __init__(self, encoder, embedding, centroids, metric) -> None:
        super().__init__()
        self.encoder = encoder
        self.embedding = embedding
        self.centroids = nn.Parameter(centroids)
        self.metric = metric

    def dist_to_centroids(self, x):
        dists = torch.zeros(x.shape[0], self.centroids.shape[0])

        for i in range(x.shape[0]):
            for j in range(self.centroids.shape[0]):
                dists[i][j] = self.metric(x[i], self.centroids[j])
        return dists

    def students_t_distribution_kernel(self, x, alpha=1):
        num = torch.pow((1 + x / alpha), -(alpha + 1) / 2)
        denom = num.sum(dim=1).reshape(-1, 1).repeat(1, self.centroids.shape[0])
        return num / denom

    def target_distribution(self, Q):
        F = Q.sum(dim=0)
        num = (Q**2) / F
        denom = num.sum(dim=1).reshape(-1, 1).repeat(1, Q.shape[-1])
        return num / denom

    def forward(self, x):
        h = self.encoder(x)
        embedding_results = self.embedding(h)
        D = self.dist_to_centroids(embedding_results[0])
        Q = self.students_t_distribution_kernel(D)
        P = self.target_distribution(Q)
        return embedding_results, Q, P
