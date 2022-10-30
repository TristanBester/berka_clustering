import torch.nn as nn


class IdentityEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x, None, None
