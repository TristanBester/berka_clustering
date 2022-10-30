import torch.nn as nn


class ClusterModel(nn.Module):
    def __init__(self, cluster_layer, decoder, loss_handler) -> None:
        super().__init__()
        self.cluster_layer = cluster_layer
        self.decoder = decoder
        self.loss_handler = loss_handler

    def forward(self, x):
        return self.loss_handler(self.cluster_layer, self.decoder, x)
