import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters=[64, 64, 64], kernels=[8, 5, 3]):
        super().__init__()
        self.cnn_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters[0],
            kernel_size=kernels[0],
            padding="same",
        )
        self.bn_1 = nn.BatchNorm1d(num_features=filters[0])

        self.cnn_2 = nn.Conv1d(
            in_channels=filters[0],
            out_channels=filters[1],
            kernel_size=kernels[1],
            padding="same",
        )
        self.bn_2 = nn.BatchNorm1d(num_features=filters[1])

        self.cnn_3 = nn.Conv1d(
            in_channels=filters[1],
            out_channels=filters[2],
            kernel_size=kernels[2],
            padding="same",
        )
        self.bn_3 = nn.BatchNorm1d(num_features=filters[2])

        self.short_cut = nn.Conv1d(
            in_channels=in_channels,
            out_channels=filters[2],
            kernel_size=[1],
            padding="same",
        )

    def forward(self, x):
        i = x

        x = self.cnn_1(x)
        x = F.relu(self.bn_1(x))

        x = self.cnn_2(x)
        x = F.relu(self.bn_2(x))

        x = self.cnn_3(x)
        x = self.bn_3(x)

        i = self.short_cut(i)
        x = i + x
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, seq_len, embedding_dim, pooling_kernel) -> None:
        super().__init__()
        self.res_block_1 = ResidualBlock(in_channels=in_channels)
        self.res_block_2 = ResidualBlock(in_channels=64)
        self.res_block_3 = ResidualBlock(in_channels=64)
        self.pooling = nn.AvgPool1d(kernel_size=pooling_kernel)
        self.embedding = nn.Linear(
            in_features=64 * (seq_len // pooling_kernel), out_features=embedding_dim
        )

    def forward(self, x):
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)

        x = self.pooling(x)
        x = x.flatten(start_dim=1)
        return x


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.embedding = nn.Linear(in_features=embedding_dim, out_features=64 * seq_len)
        self.res_block_1 = ResidualBlock(in_channels=64)
        self.res_block_2 = ResidualBlock(in_channels=64)
        self.res_block_3 = ResidualBlock(in_channels=64, filters=[64, 64, 1])

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.embedding(x))
        x = x.reshape(batch_size, 64, self.seq_len)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        return x


class ResNetVAE(nn.Module):
    def __init__(self, in_channels, seq_len, embedding_dim, pooling_kernel) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            seq_len=seq_len,
            embedding_dim=embedding_dim,
            pooling_kernel=pooling_kernel,
        )
        self.decoder = Decoder(seq_len=seq_len, embedding_dim=embedding_dim)

        self.z_mean = nn.Linear(64 * (seq_len // pooling_kernel), embedding_dim)
        self.z_log_var = nn.Linear(64 * (seq_len // pooling_kernel), embedding_dim)

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
