import torch.nn as nn
import torch.nn.functional as F


class FCNNDecoder(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(in_features=embedding_dim, out_features=2000)
        self.fc_2 = nn.Linear(in_features=2000, out_features=500)
        self.fc_3 = nn.Linear(in_features=500, out_features=seq_len)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        return F.relu(self.fc_3(x)).unsqueeze(1)

    def layerwise_forward(self, x):
        l_1 = F.relu(self.fc_1(x))
        l_2 = F.relu(self.fc_2(l_1))
        l_3 = F.relu(self.fc_3(l_2))
        return [
            l_3.unsqueeze(1),
            l_2.unsqueeze(1),
            l_1.unsqueeze(1),
        ]
