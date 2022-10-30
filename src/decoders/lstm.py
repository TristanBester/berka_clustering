import torch
import torch.nn as nn


class LSTMDecoder(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.lstm_1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=50,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.lstm_2 = nn.LSTM(
            input_size=50 * 2,
            hidden_size=1,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        # Upsample latent representation
        x = x.unsqueeze(1)
        x = x.repeat(1, self.seq_len, 1)

        outputs, (_, _) = self.lstm_1(x)
        outputs, (_, _) = self.lstm_2(outputs)

        outputs = torch.sum(outputs, dim=-1).unsqueeze(-1)
        return outputs.permute(0, 2, 1)
