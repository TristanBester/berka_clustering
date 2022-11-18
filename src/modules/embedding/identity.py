import torch.nn as nn


class IdentityEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, device=None):
        return x, None, None
