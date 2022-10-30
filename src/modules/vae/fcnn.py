import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, seq_len) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(in_features=seq_len, out_features=500)
        self.fc_2 = nn.Linear(in_features=500, out_features=500)
        self.fc_3 = nn.Linear(in_features=500, out_features=2000)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        return x


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(in_features=embedding_dim, out_features=2000)
        self.fc_2 = nn.Linear(in_features=2000, out_features=500)
        self.fc_3 = nn.Linear(in_features=500, out_features=seq_len)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return F.relu(self.fc_3(x))


class FCNNVAE(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()
        self.encoder = Encoder(seq_len=seq_len)
        self.decoder = Decoder(seq_len=seq_len, embedding_dim=embedding_dim)
        self.z_mean = nn.Linear(2000, embedding_dim)
        self.z_log_var = nn.Linear(2000, embedding_dim)

    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.shape)
        z = z_mu + eps * torch.exp(z_log_var / 2)
        return z

    def forward(self, x):
        l = self.encoder(x)
        z_mean = self.z_mean(l)
        z_log_var = self.z_log_var(l)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return decoded, z_mean, z_log_var
