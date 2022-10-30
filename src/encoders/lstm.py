import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()

        if embedding_dim % 2 != 0:
            raise ValueError("'embedding_dim' must be even.")

        self.lstm_1 = nn.LSTM(
            input_size=1,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_2 = nn.LSTM(
            input_size=50 * 2,
            hidden_size=embedding_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.output_dim = embedding_dim

    def forward(self, x):
        x = x.permute(0, 2, 1)

        outputs, (_, _) = self.lstm_1(x)
        _, (h_n, _) = self.lstm_2(outputs)

        # Concatenate forward and reverse passes
        h_n = h_n.permute(1, 0, 2)
        h_n = torch.flatten(h_n, start_dim=1)
        return h_n
