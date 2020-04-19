import torch
import numpy as np


def event_flow_warp(score_ori, warp=1, max_val=1):
    """ Warping for event flow module
    """
    if isinstance(score_ori, torch.Tensor):
        is_tensor = True
        score_ori = score_ori.detach().cpu().numpy()
    else:
        is_tensor = False

    score = max_val - score_ori
    row, col = score.shape
    distance_mat = np.zeros((row + 1, col + 1))
    distance_mat[0, 1:] = np.inf
    distance_mat[1:, 0] = np.inf
    sub_dmat = distance_mat[1:, 1:]
    distance_mat[1:, 1:] = score

    for i in range(row):
        for j in range(col):
            min_list = [distance_mat[i, j]]
            for k in range(1, warp + 1):
                min_list += [
                    distance_mat[min(i + k, row - 1), j],
                    distance_mat[i, min(j + k, col - 1)]
                ]
            sub_dmat[i, j] += min(min_list)

    if row == 1:
        path = [[0] * col, range(col)]
    elif col == 1:
        path = [range(row), [0] * row]
    else:
        path = _traceback(distance_mat)

    if is_tensor:
        return path
    else:
        return tuple(path)


def _traceback(mat):
    i, j = np.array(mat.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = np.argmin((mat[i, j], mat[i, j + 1], mat[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
    return [p, q]