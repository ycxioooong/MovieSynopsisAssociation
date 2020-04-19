import numpy as np
import torch
from scipy.spatial.distance import cdist

__all__ = ['recall_1d', 'recall_2d']


def recall_1d(score_matrix, k=(1, )):
    """ A M * N matrix with M[:, 0] as gt. Score matrix: the higher the better
    """
    if isinstance(score_matrix, torch.Tensor):
        return recall_1d_tensor(score_matrix, k)
    elif isinstance(score_matrix, np.ndarray):
        return recall_1d_np(score_matrix, k)


def recall_1d_np(score_matrix, k):
    inds = np.argsort(-score_matrix, axis=1)
    ranks = np.where(inds == 0)[1]
    top_recall = [(ranks < k_).sum() / float(score_matrix.shape[0])
                  for k_ in k]
    if len(top_recall) == 1:
        top_recall = top_recall[0]
    medr = np.median(ranks) + 1
    meanr = np.mean(ranks) + 1
    return top_recall, medr, meanr


def recall_1d_tensor(score_matrix, k):
    _, inds = torch.sort(score_matrix, descending=True)
    ranks = torch.nonzero(inds == 0)[:, 1]
    top_recall = [(ranks < k_).sum().float() / float(score_matrix.shape[0])
                  for k_ in k]
    if len(top_recall) == 1:
        top_recall = top_recall[0]
    medr = torch.median(ranks.float()) + 1
    meanr = torch.mean(ranks.float()) + 1
    return top_recall, medr, meanr


def recall_2d(query_mat=None,
              source_mat=None,
              score=None,
              norm=False,
              k=(1, ),
              return_meta=False):
    """ Calculate similarity scores between query matrix and source matrix, and
    return the rank recall.
    Args:
        query_mat: 2d numpy array
        source_mat: 2d numpy array with the same shape as query mat
        norm: whether or not norm the matrix
        k: top k
        return_mata: whether return the meta infos
    
    Return:
        top / medium / mean recalls.
        ground-truth ranks [if return meta]
        top1 indice [if return meta]
        all indice [if return meta]
    """
    if score is None:
        if isinstance(query_mat, torch.Tensor):
            query_mat = query_mat.detach().cpu().numpy()
        if isinstance(source_mat, torch.Tensor):
            source_mat = source_mat.detach().cpu().numpy()

        assert query_mat.shape == source_mat.shape
        num_vec = query_mat.shape[0]
        if norm:
            distances = cdist(query_mat, source_mat, 'cosine')
        else:
            distances = 1.0 - np.dot(query_mat, source_mat.T)
    else:
        num_vec = score.shape[0]
        assert score.shape[0] == score.shape[1]
        distances = 1.0 - score

    inds = np.argsort(distances, axis=1)

    tmp = inds - np.arange(num_vec)[..., None]
    ranks = np.where(tmp == 0)[1]

    top_recall = [(ranks < k_).sum() / float(num_vec) for k_ in k]
    medr = np.median(ranks) + 1
    meanr = np.mean(ranks) + 1

    if return_meta:
        return top_recall, medr, meanr, ranks, inds
    else:
        return top_recall, medr, meanr
