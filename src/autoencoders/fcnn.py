import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(in_features=seq_len, out_features=500)
        self.fc_2 = nn.Linear(in_features=500, out_features=500)
        self.fc_3 = nn.Linear(in_features=500, out_features=2000)
        self.embedding = nn.Linear(in_features=2000, out_features=embedding_dim)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        return F.relu(self.embedding(x)).squeeze(1)


class Decoder(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(in_features=embedding_dim, out_features=2000)
        self.fc_2 = nn.Linear(in_features=2000, out_features=500)
        self.fc_3 = nn.Linear(in_features=500, out_features=500)
        self.fc_4 = nn.Linear(in_features=500, out_features=seq_len)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        return F.relu(self.fc_4(x)).unsqueeze(1)


class FCNNAutoencoder(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()
        self.encoder = Encoder(seq_len=seq_len, embedding_dim=embedding_dim)
        self.decoder = Decoder(seq_len=seq_len, embedding_dim=embedding_dim)

    def forward(self, x):
        l = self.encoder(x)
        return self.decoder(l)
