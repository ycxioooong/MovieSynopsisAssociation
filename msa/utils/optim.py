import numpy as np
from msa.ops import event_flow_warp, fast_bimatch


def efw_get_Y(inds, shape):
    Y = np.zeros(shape, dtype=np.float32)
    Y[inds] = 1
    return Y


def efw_wrapper(data, i, j):
    inds = event_flow_warp(data)
    Y = efw_get_Y(inds, data.shape)
    return i, j, Y


def bm_wrapper_score(data, i, j):
    _, s = fast_bimatch(data.T)
    return i, j, s


def bm_get_Y(inds, shape):
    Y = np.zeros(shape, dtype=np.float32)
    for i in range(Y.shape[1]):
        if inds[i] != -1:
            Y[inds[i], i] = 1
    return Y


def bm_wrapper(data, i, j):
    inds, _ = fast_bimatch(data.T + 1)
    Y = bm_get_Y(list(inds), data.shape)
    return i, j, Y
