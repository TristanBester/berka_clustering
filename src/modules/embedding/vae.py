import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEEmbedding(nn.Module):
    def __init__(self, in_features, embedding_dim) -> None:
        super().__init__()
        self.z_mean_embedding = nn.Linear(
            in_features=in_features, out_features=embedding_dim
        )
        self.z_log_var_embedding = nn.Linear(
            in_features=in_features, out_features=embedding_dim
        )

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.shape)
        z = z_mu + eps * torch.exp(z_log_var / 2)
        return z.squeeze(1)

    def forward(self, x):
        z_mean = F.relu(self.z_mean_embedding(x))
        z_log_var = F.relu(self.z_log_var_embedding(x))
        z = self.reparameterize(z_mean, z_log_var)
        return z, z_mean, z_log_var
