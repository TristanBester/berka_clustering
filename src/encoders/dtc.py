import torch.nn as nn
import torch.nn.functional as F


class DTCEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        cnn_channels,
        cnn_kernel,
        cnn_stride,
        mp_kernel,
        mp_stride,
        embedding_dim,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.cnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=cnn_kernel,
            stride=cnn_stride,
        )
        self.max_pool = nn.MaxPool1d(
            kernel_size=mp_kernel,
            stride=mp_stride,
        )

        self.lstm_layer_one = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=embedding_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        self.lstm_layer_two = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=1,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, x):
        x = F.leaky_relu(self.cnn(x))
        x = self.max_pool(x)

        x = x.permute(0, 2, 1)
        x, (_, _) = self.lstm_layer_one(x)
        x = x[:, :, : self.embedding_dim] + x[:, :, : self.embedding_dim]

        x, (_, _) = self.lstm_layer_two(x)
        x = (x[:, :, 0] + x[:, :, 1]).unsqueeze(-1)
        x = x.permute(0, 2, 1)
        return x
