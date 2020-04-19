import torch
import numpy as np
import multiprocessing as mp

from scipy.linalg import block_diag
from gurobipy import *

from .graph import (graph_pruning, reduce_syn_cast_cast,
                    reduce_syn_cast_action, reduce_video)
from .misc import acc_list, clamp_cdist, fast_bimatch


def gen_kernel_matrix(cast_score, action_score, vcc, vca, scc, sca, aw=0.1):
    m1, n1 = cast_score.shape
    m2, n2 = action_score.shape

    if isinstance(cast_score, torch.Tensor):
        sim = cast_score.new_zeros((m1 + m2, n1 + n2))
        sim[:m1, :n1] = cast_score
        sim[m1:, n1:] = action_score * aw
        sim.clamp(min=0)
        M = torch.diag(sim.flatten())
    else:
        sim = block_diag(cast_score, action_score * aw)
        sim[sim < 0] = 0
        M = np.diag(sim.flatten())

    for i, j in vcc:
        for a, b in scc:
            M[i * (n1 + n2) + a, j * (n1 + n2) + b] = 0.5 * (
                sim[i, a] + sim[j, b])

    for i, j in vca:
        for a, b in sca:
            M[i * (n1 + n2) + a, (m1 + j) * (n1 + n2) + n1 + b] = 0.5 * (
                sim[i, a] + sim[m1 + j, n1 + b])

    return M


def get_graph(loader, level, add_temporal=True):
    assert level in ['para', 'sent']
    result_vid = []
    result_syn = []
    for meta in loader:
        vcc = meta['vcc']
        vct = meta['vct']
        vco = meta['vco']
        vca = meta['vca']
        vao = meta['vao']

        scc = meta['scc']
        sco = meta['sco']
        sca = meta['sca']
        sao = meta['sao']

        if level == 'para':
            vcc = reduce_video(vcc, vco[0], vco[0])
            if vct is not None:
                vct = reduce_video(vct, vco[0], vco[0])
            if add_temporal:
                vcc += vct
            vca = reduce_video(vca, vco[0], vao[0])
            scc = reduce_syn_cast_cast(scc, sco)
            sca = reduce_syn_cast_action(sca, sco, sao)
            result_vid.append(
                dict(
                    video_cast_cast=vcc,
                    video_cast_temporal=vct,
                    video_cast_action=vca,
                    mid=meta['mid'],
                    syn_id=meta['syn_id']))
            result_syn.append(
                dict(
                    syn_cast_cast=scc,
                    syn_cast_action=sca,
                    mid=meta['mid'],
                    syn_id=meta['syn_id']))
        else:
            result_vid.append(dict(vcc=vcc, vco=vco, vca=vca, vao=vao))
            result_syn.append(dict(scc=scc, sco=sco, sca=sca, sao=sao))
    return result_vid, result_syn


def mp_graph_match_core(cast_clips,
                        cast_syns,
                        cast_clens,
                        cast_slens,
                        action_clips,
                        action_syns,
                        action_clens,
                        action_slens,
                        video_graphs,
                        syn_graphs,
                        mask,
                        argmax=2,
                        degree=2,
                        aw=0.1,
                        max_mp_shape=100000,
                        max_gm_shape=3000,
                        nproc=24):

    nquery = len(cast_clens)
    cast_clens = acc_list(cast_clens)[:-1]
    cast_slens = acc_list(cast_slens)[:-1]
    action_clens = acc_list(action_clens)[:-1]
    action_slens = acc_list(action_slens)[:-1]
    cast_clip_lst = np.split(cast_clips, cast_clens, axis=0)
    cast_syn_lst = np.split(cast_syns, cast_slens, axis=0)
    action_clip_lst = np.split(action_clips, action_clens, axis=0)
    action_syn_lst = np.split(action_syns, action_slens, axis=0)

    pool = mp.Pool(nproc)
    results = []

    for i in range(nquery):
        for j in range(nquery):
            rst = pool.apply_async(
                graph_matching_single,
                args=(i, j, cast_clip_lst[i].copy(), cast_syn_lst[j].copy(),
                      action_clip_lst[i].copy(), action_syn_lst[j].copy(),
                      video_graphs[i], syn_graphs[j], mask[i, j], argmax,
                      degree, aw, max_mp_shape, max_gm_shape))
            results.append(rst)

    pool.close()
    pool.join()
    results = [r.get() for r in results]
    score = np.empty((nquery, nquery))

    for i, j, s in results:
        score = s

    return score


def graph_matching_single(i, j, *args, **kwargs):
    try:
        s = _graph_matching(*args, **kwargs)
    except Exception as e:
        s = 0
        print(e)
    return i, j, s


def _graph_matching(cast_clip,
                    cast_syn,
                    action_clip,
                    action_syn,
                    vg,
                    sg,
                    mask,
                    argmax=2,
                    degree=2,
                    aw=0.1,
                    max_mp_shape=100000,
                    max_gm_shape=3000):
    if not mask:
        return 0

    cast_score = clamp_cdist(cast_clip, cast_syn)
    init_sln_c, _sc = fast_bimatch(cast_score.T)
    init_sln_c = list(init_sln_c)
    action_score = clamp_cdist(action_clip, action_syn)
    init_sln_a, _sa = fast_bimatch(action_score.T)
    init_sln_a = list(init_sln_a)

    vcc = vg['video_cast_cast']
    vca = vg['video_cast_action']
    scc = sg['syn_cast_cast']
    sca = sg['syn_cast_action']
    summ = cast_score.shape[1] + action_score.shape[1]
    if aw == 0:
        summ = cast_score.shape[1]
        vca = []
        sca = []

    (cast_score, action_score, _, _, vcc, vca, scc, sca,
     init_Y) = graph_pruning(
         cast_score,
         action_score,
         init_sln_c,
         init_sln_a,
         vcc,
         vca,
         scc,
         sca,
         argmax=argmax,
         degree=degree,
         max_mp_shape=max_mp_shape,
         max_gm_shape=max_gm_shape)
    M_p = gen_kernel_matrix(
        cast_score, action_score, vcc, vca, scc, sca, aw=aw)

    if M_p.shape[0] < max_mp_shape:
        t_ = np.dot(M_p, init_Y.T)
        score_init = np.dot(init_Y, t_) / max(1, summ)
    else:
        score_init = (_sc + _sa) / max(1, summ)

    if (M_p.shape[0] >= max_gm_shape
            or ((vcc == [] and vca == []) or (scc == [] and sca == []))):
        score_final = score_init
    else:
        m1, n1 = cast_score.shape
        m2, n2 = action_score.shape
        v = qap(M_p, m1, n1, m2, n2, Y=init_Y)
        t_ = np.dot(M_p, v.T)
        score_final = np.dot(v, t_) / max(1, summ)
    return score_final


def qap(M, m1, n1, m2, n2, Y=None):

    m = Model('qap')
    m.setParam('OutputFlag', False)

    x = m.addVars((m1 + m2) * (n1 + n2), vtype=GRB.BINARY, name='x')
    obj = quicksum(
        quicksum(x[i] * x[j] * M[i, j] for j in range((m1 + m2) * (n1 + n2)))
        for i in range((m1 + m2) * (n1 + n2)))
    m.setObjective(obj, GRB.MAXIMIZE)

    cast_row_starts = (np.arange(m1) * (n1 + n2)).tolist()
    action_row_starts = (np.arange(m1, m1 + m2) * (n1 + n2)).tolist()

    cast_cols = list(range(n1))
    action_cols = list(range(n1, n1 + n2))

    m.addConstrs((quicksum(x[i + j] for i in cast_row_starts) <= 1
                  for j in cast_cols), 'c0')
    m.addConstrs((quicksum(x[i + j] for j in cast_cols) <= 1
                  for i in cast_row_starts), 'c1')
    m.addConstrs((quicksum(x[i + j] for i in cast_row_starts) == 0
                  for j in action_cols), 'c2')

    m.addConstrs((quicksum(x[i + j] for i in action_row_starts) <= 1
                  for j in action_cols), 'c3')
    m.addConstrs((quicksum(x[i + j] for j in action_cols) <= 1
                  for i in action_row_starts), 'c4')
    m.addConstrs((quicksum(x[i + j] for i in action_row_starts) == 0
                  for j in cast_cols), 'c5')

    if Y is not None:
        for y, v in zip(Y, m.getVars()):
            v.start = y
        m.update()

    m.optimize()

    res = [v.X for v in m.getVars()]
    return np.array(res).astype(int)
