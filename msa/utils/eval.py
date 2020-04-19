import multiprocessing as mp
import numpy as np
from .misc import acc_list
from .math import clamp_cdist
from .optim import efw_wrapper, bm_wrapper_score
from msa.ops import fast_bimatch


def get_score_with_score_return(func,
                                seq_score,
                                clips,
                                syns,
                                clip_len,
                                syn_len,
                                nproc=32):

    pool = mp.Pool(nproc)
    norm = np.array(syn_len)

    score = np.empty((len(clip_len), len(syn_len)))
    clip_len = acc_list(clip_len)[:-1]
    syn_len = acc_list(syn_len)[:-1]

    row_tensors = np.split(seq_score, clip_len, axis=0)

    results = []
    for i, row_tensor in enumerate(row_tensors):
        datas = np.split(row_tensor, syn_len, axis=1)
        for j, data in enumerate(datas):
            result = pool.apply_async(func, args=(data, i, j))
            results.append(result)

    pool.close()
    pool.join()

    results = [r.get() for r in results]
    for i, j, s in results:
        score[i, j] = s

    score_normed = score / norm

    return score, score_normed, None


def get_score_with_matrix_return(func, seq_score, clips, syns, clip_len,
                                 syn_len):
    pool = mp.Pool(32)
    norm = np.array(syn_len)
    score = np.empty((len(clip_len), len(syn_len)))
    clip_len = acc_list(clip_len)[:-1]
    syn_len = acc_list(syn_len)[:-1]
    row_tensors = np.split(seq_score, clip_len, axis=0)

    results = []
    for i, row_tensor in enumerate(row_tensors):
        datas = np.split(row_tensor, syn_len, axis=1)
        for j, data in enumerate(datas):
            result = pool.apply_async(func, args=(data, i, j))
            results.append(result)

    pool.close()
    pool.join()

    results = [r.get() for r in results]
    ind_dict = dict()
    for i, j, Y in results:
        ind_dict['{}_{}'.format(i, j)] = Y

    clip_lst = np.split(clips, clip_len, axis=0)
    syn_lst = np.split(syns, syn_len, axis=0)
    for i, clip in enumerate(clip_lst):
        for j, syn in enumerate(syn_lst):
            Y = ind_dict['{}_{}'.format(i, j)]
            s = clamp_cdist(syn, np.dot(Y.T, clip))
            score[i, j] = s.diagonal().mean()

    score_normed = score / norm

    return score, score_normed, ind_dict


# ====================================================


def get_score_efw(seq_score, clips, syns, clip_len, syn_len):
    return get_score_with_matrix_return(efw_wrapper, seq_score, clips, syns,
                                        clip_len, syn_len)


def get_score_bimatch(seq_score, clips, syns, clip_len, syn_len):
    return get_score_with_score_return(bm_wrapper_score, seq_score, clips,
                                       syns, clip_len, syn_len)


def get_score_basic(clips, syns, clip_len, syn_len):

    clip_len = acc_list(clip_len)[:-1]
    syn_len = acc_list(syn_len)[:-1]
    clip_lst = np.split(clips, clip_len, axis=0)
    syn_lst = np.split(syns, syn_len, axis=0)
    clips = np.array([c.mean(axis=0) for c in clip_lst])
    syns = np.array([s.mean(axis=0) for s in syn_lst])
    score = clamp_cdist(clips, syns)

    return score
