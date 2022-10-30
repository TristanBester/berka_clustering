import torch
import torch.nn as nn


class Encoder(nn.Module):
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

    def forward(self, x):
        x = x.permute(0, 2, 1)

        outputs, (_, _) = self.lstm_1(x)
        _, (h_n, c_n) = self.lstm_2(outputs)

        # Concatenate forward and reverse passes
        h_n = h_n.permute(1, 0, 2)
        h_n = torch.flatten(h_n, start_dim=1).unsqueeze(1)
        return h_n


class Decoder(nn.Module):
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
        x = x.repeat(1, self.seq_len, 1)

        outputs, (_, _) = self.lstm_1(x)
        outputs, (_, _) = self.lstm_2(outputs)

        outputs = torch.sum(outputs, dim=-1).unsqueeze(-1)
        return outputs.permute(0, 2, 1)


class BiLSTMVAE(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()
        self.encoder = Encoder(embedding_dim=embedding_dim)
        self.decoder = Decoder(seq_len=seq_len, embedding_dim=embedding_dim)
        self.z_mean = nn.Linear(embedding_dim, embedding_dim)
        self.z_log_var = nn.Linear(embedding_dim, embedding_dim)

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
