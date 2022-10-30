import torch


def correlation_based_similarity(x, y):
    t = torch.vstack((x, y))
    p = torch.corrcoef(t)[0][1]
    return torch.sqrt(2 * (1 - p))
