import torch


def euclidean_distance(x, y):
    return torch.sum((x - y) ** 2)
