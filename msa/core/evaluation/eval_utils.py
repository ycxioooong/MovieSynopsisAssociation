import torch
from msa.utils import (fast_bimatch, graph_pruning, gen_kernel_matrix,
                       efw_wrapper, bm_wrapper)
# import multiprocessing as mp
import torch.nn.functional as F
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

__all__ = [
    'get_score_cosine_similarity', 'get_score_gm_approx', 'get_score_efw',
    'get_score_bm'
]


def gm_approx(score_cast,
              score_action,
              vg,
              sg,
              i,
              j,
              aw=0.1,
              argmax=0,
              degree=0):
    cast_score = score_cast.detach().cpu().numpy()
    action_score = score_action.detach().cpu().numpy()
    summ = cast_score.shape[1] + action_score.shape[1]

    init_sln_c, _sc = fast_bimatch(cast_score.T)
    init_sln_c = list(init_sln_c)
    init_sln_a, _sa = fast_bimatch(action_score.T)
    init_sln_a = list(init_sln_a)

    vcc = vg['video_cast_cast']
    vca = vg['video_cast_action']
    scc = sg['syn_cast_cast']
    sca = sg['syn_cast_action']

    (_, _, left_cast, left_action, vcc, vca, scc, sca, Y) = graph_pruning(
        cast_score,
        action_score,
        init_sln_c,
        init_sln_a,
        vcc,
        vca,
        scc,
        sca,
        argmax=argmax,
        degree=degree)

    M = gen_kernel_matrix(
        score_cast[left_cast],
        score_action[left_action],
        vcc,
        vca,
        scc,
        sca,
        aw=aw)
    Y = M.new_tensor(Y)[None, ...]
    t_ = torch.mm(M, Y.t())
    s = torch.mm(Y, t_) / max(1, summ)
    return i, j, s[0, 0]


# tensor version of get scores
def get_score_cosine_similarity(clips, syns, clip_len, syn_len):
    clip_lst = torch.split(clips, clip_len, dim=0)
    syn_lst = torch.split(syns, syn_len, dim=0)
    clip_embed = [c.mean(dim=0, keepdim=True) for c in clip_lst]
    syn_embed = [s.mean(dim=0, keepdim=True) for s in syn_lst]
    ce = torch.cat(clip_embed, 0)
    se = torch.cat(syn_embed, 0)
    s = torch.mm(F.normalize(ce), F.normalize(se).t())
    return s


def get_score_efw(seq_score, clips, syns, clip_len, syn_len, nproc=24):
    if nproc == 1:
        return _get_score_mat_match(seq_score, clips, syns, clip_len, syn_len,
                                    efw_wrapper)
    else:
        return _get_score_mat_match_mp(seq_score, clips, syns, clip_len,
                                       syn_len, efw_wrapper, nproc)


def get_score_bm(seq_score, clips, syns, clip_len, syn_len, nproc=24):
    if nproc == 1:
        return _get_score_mat_match(seq_score, clips, syns, clip_len, syn_len,
                                    bm_wrapper)
    else:
        return _get_score_mat_match_mp(seq_score, clips, syns, clip_len,
                                       syn_len, bm_wrapper, nproc)


def _get_score_mat_match(seq_score, clips, syns, clip_len, syn_len, func):
    score = seq_score.new_empty((len(clip_len), len(syn_len)))
    row_tensors = torch.split(seq_score, clip_len, dim=0)
    clip_lst = torch.split(clips, clip_len, dim=0)
    syn_lst = torch.split(syns, syn_len, dim=0)

    for i, row_tensor in enumerate(row_tensors):
        datas = torch.split(row_tensor, syn_len, dim=1)
        for j, data in enumerate(datas):
            _, _, Y = func(data.detach().cpu().numpy(), i, j)
            score[i, j] = F.cosine_similarity(syn_lst[j],
                                              torch.mm(Y.t(),
                                                       clip_lst[i])).mean()
    return score


def _get_score_mat_match_mp(seq_score,
                            clips,
                            syns,
                            clip_len,
                            syn_len,
                            func,
                            nproc=24):
    score = seq_score.new_empty((len(clip_len), len(syn_len)))
    row_tensors = torch.split(seq_score, clip_len, dim=0)

    pool = mp.Pool(nproc)

    results = []
    for i, row_tensor in enumerate(row_tensors):
        datas = torch.split(row_tensor, syn_len, dim=1)
        for j, data in enumerate(datas):
            result = pool.apply_async(
                func, args=(data.detach().cpu().numpy(), i, j))
            results.append(result)

    pool.close()
    pool.join()

    results = [r.get() for r in results]
    ind_dict = dict()
    for i, j, Y in results:
        ind_dict['{}_{}'.format(i, j)] = Y

    clip_lst = torch.split(clips, clip_len, dim=0)
    syn_lst = torch.split(syns, syn_len, dim=0)
    for i, clip in enumerate(clip_lst):
        for j, syn in enumerate(syn_lst):
            Y = ind_dict['{}_{}'.format(i, j)]
            Y = syn.new_tensor(Y)
            score[i, j] = F.cosine_similarity(syn, torch.mm(Y.t(),
                                                            clip)).mean()

    return score


def get_score_gm_approx(seq_score_cast,
                        clip_len_cast,
                        syn_len_cast,
                        seq_score_action,
                        clip_len_action,
                        syn_len_action,
                        vgs,
                        sgs,
                        aw=0.1):
    score = seq_score_cast.new_empty((len(clip_len_cast), len(syn_len_cast)))
    row_tensors_cast = torch.split(seq_score_cast, clip_len_cast, dim=0)
    row_tensors_action = torch.split(seq_score_action, clip_len_action, dim=0)

    for i, (row_tensor_cast, row_tensor_action) in enumerate(
            zip(row_tensors_cast, row_tensors_action)):
        datas_cast = torch.split(row_tensor_cast, syn_len_cast, dim=1)
        datas_action = torch.split(row_tensor_action, syn_len_action, dim=1)

        for j, (data_cast,
                data_action) in enumerate(zip(datas_cast, datas_action)):
            _, _, _score = gm_approx(data_cast, data_action, vgs[i], sgs[j], i,
                                     j, aw)
            score[i, j] = _score

    return score
