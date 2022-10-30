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


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels, seq_len, embedding_dim, pooling_kernel) -> None:
        super().__init__()
        self.res_block_1 = ResidualBlock(in_channels=in_channels)
        self.res_block_2 = ResidualBlock(in_channels=64)
        self.res_block_3 = ResidualBlock(in_channels=64)
        self.pooling = nn.AvgPool1d(kernel_size=pooling_kernel)
        self.output_dim = 64 * (seq_len // pooling_kernel)

    def forward(self, x):
        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)

        x = self.pooling(x)
        x = x.flatten(start_dim=1)
        return x

    def layerwise_forward(self, x):
        l_1 = self.res_block_1(x)
        l_2 = self.res_block_2(l_1)
        l_3 = self.res_block_3(l_2)

        l = self.pooling(l_3)
        l = l.flatten(start_dim=1)
        return [x, l_1, l_2, l_3], l
