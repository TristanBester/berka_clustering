import torch

from .euclidean import euclidean_distance


def _complexity_estimate(x):
    x_back_shift = x[:-1]
    x_forward_shift = x[1:]
    return torch.sqrt(torch.sum((x_forward_shift - x_back_shift) ** 2))


def _complexitity_factor(x, y):
    ce = torch.tensor([_complexity_estimate(x), _complexity_estimate(y)])
    return torch.max(ce) / (torch.min(ce) + 1e-5)


def complexity_invariant_similarity(x, y):
    ed = euclidean_distance(x, y)
    cf = _complexitity_factor(x, y)
    return ed * cf
