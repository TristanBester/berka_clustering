import torch.nn as nn
import torch.nn.functional as F


class FCNNEncoder(nn.Module):
    def __init__(self, seq_len, embedding_dim) -> None:
        super().__init__()
        self.fc_1 = nn.Linear(in_features=seq_len, out_features=500)
        self.fc_2 = nn.Linear(in_features=500, out_features=500)
        self.fc_3 = nn.Linear(in_features=500, out_features=2000)
        self.output_dim = 2000

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = x.flatten(start_dim=1)
        return x

    def layerwise_forward(self, x):
        l_1 = F.relu(self.fc_1(x))
        l_2 = F.relu(self.fc_2(l_1))
        l_3 = F.relu(self.fc_3(l_2))
        h = l_3.flatten(start_dim=1)
        return [x, l_2, l_3], h
