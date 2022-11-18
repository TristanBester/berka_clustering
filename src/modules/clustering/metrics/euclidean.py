import numpy as np
import torch


def euclidean_distance_np(x, y):
    arr = (x-y)**2
    return np.sum(arr)
 
def euclidean_distance_torch(x, y):
    arr = (x-y)**2
    return torch.sum(arr)
 