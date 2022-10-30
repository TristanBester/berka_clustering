import torch.nn as nn
import torch.nn.functional as F


class VectorEmbedding(nn.Module):
    def __init__(self, in_features, embedding_dim) -> None:
        super().__init__()
        self.embedding = nn.Linear(in_features=in_features, out_features=embedding_dim)

    def forward(self, x):
        return F.relu(self.embedding(x)).squeeze(1), None, None
