import numpy as np
import torch


def correlation_based_similarity_np(x, y):
    t = np.vstack((x,y))
    p = np.corrcoef(t)[0][1]
    return np.sqrt(2 * (1-p))

def correlation_based_similarity_torch(x, y):
    t = torch.vstack((x,y))
    p = torch.corrcoef(t)[0][1]
    return torch.sqrt(2 * (1-p))
