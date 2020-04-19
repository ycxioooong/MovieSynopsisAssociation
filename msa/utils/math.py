import torch
from numpy import linalg as LA
import numpy as np

__all__ = ['l2norm', 'clamp_cdist']


def l2norm(X, dim=1, eps=1e-12):
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt().clamp(min=eps)
    X = torch.div(X, norm.expand_as(X))
    return X


def clamp_cdist(x, y, eps=1e-12):
    """ Equal to scipy.spatial.distance.cdist(x, y, 'cosine') 
        with bug caused by zero denominator fixed.
    """

    x = x / np.maximum(LA.norm(x, axis=1), eps)[..., None]
    y = y / np.maximum(LA.norm(y, axis=1), eps)[..., None]
    return np.dot(x, y.T)
