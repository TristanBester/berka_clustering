import numpy as np
import torch

from .euclidean import euclidean_distance_np, euclidean_distance_torch


def _complexity_estimate_np(x):
    x_back_shift = x[:-1]
    x_forward_shift = x[1:]
    return np.sqrt(np.sum((x_forward_shift - x_back_shift) ** 2))


def _complexitity_factor_np(x, y):
    ce = np.array([_complexity_estimate_np(x), _complexity_estimate_np(y)])
    return np.max(ce) / (np.min(ce) + 1e-5)


def complexity_invariant_similarity_np(x, y):
    ed = euclidean_distance_np(x, y)
    cf = _complexitity_factor_np(x, y)
    return ed * cf


def _complexity_estimate_torch(x):
    x_back_shift = x[:-1]
    x_forward_shift = x[1:]
    return torch.sqrt(torch.sum((x_forward_shift - x_back_shift) ** 2))


def _complexitity_factor_torch(x, y):
    ce = torch.tensor([_complexity_estimate_torch(x), _complexity_estimate_torch(y)])
    return torch.max(ce) / (torch.min(ce) + 1e-5)


def complexity_invariant_similarity_torch(x, y):
    ed = euclidean_distance_torch(x, y)
    cf = _complexitity_factor_torch(x, y)
    return ed * cf
