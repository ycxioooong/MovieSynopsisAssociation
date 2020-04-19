import numpy as np
from scipy.linalg import block_diag


def reduce_syn_cast_cast(cast_cast, cast_offset):
    res = []
    for cc, offset in zip(cast_cast, cast_offset):
        if cc == []:
            continue
        res += [[item[0] + offset, item[1] + offset] for item in cc]
    return res


def reduce_syn_cast_action(cast_action, cast_offset, action_offset):
    res = []
    for ca, off_c, off_a in zip(cast_action, cast_offset, action_offset):
        if ca == []:
            continue
        res += [[item[0] + off_c, item[1] + off_a] for item in ca]
    return res


def reduce_video(graph, offset1, offset2):
    res = []
    for cc in graph:
        if cc is None:
            continue
        res += cc
    res = [[r[0] - offset1, r[1] - offset2] for r in res]
    return res


def convert_mapping_to_mat(sln, shape):
    Y = np.zeros(shape)
    for i in range(Y.shape[1]):
        if sln[i] != -1:
            Y[sln[i], i] = 1
    return Y


def graph_pruning(cast_score,
                  action_score,
                  sln_cast,
                  sln_action,
                  vcc,
                  vca,
                  scc,
                  sca,
                  argmax=2,
                  degree=1):

    # === gen adj matrix for video ===
    m1 = cast_score.shape[0]
    m2 = action_score.shape[0]
    adj = np.eye(m1 + m2)
    for i, j in vcc:
        adj[i, j] = 1
        adj[j, i] = 1
    for i, j in vca:
        adj[i, j + m1] = 1
        adj[j + m1, i] = 1

    # === pruning ===
    nodes = []
    for sln in sln_cast:
        if sln != -1:
            nodes.append(sln)
    for sln in sln_action:
        if sln != -1:
            nodes.append(sln + m1)

    # k largest
    if argmax > 0:
        if cast_score.shape[0] >= argmax:
            argmaxk = np.argpartition(cast_score.T, -argmax)[:, -argmax:]
            argmaxk = np.unique(argmaxk).tolist()
            nodes += argmaxk
        if action_score.shape[0] >= argmax:
            argmaxk = np.argpartition(action_score.T, -argmax)[:, -argmax:]
            argmaxk = (np.unique(argmaxk) + m1).tolist()
            nodes += argmaxk

    v = np.zeros((m1 + m2, 1))
    v[nodes] = 1

    for _ in range(degree):
        v = np.dot(adj, v)

    # === restore nodes and re-indexing ===
    left_cast_nodes, left_action_nodes = [], []
    cast_idx_map = dict()
    action_idx_map = dict()
    left_nodes = np.nonzero(v)[0]  # !! make sure it is sorted
    for n in left_nodes:
        if n - m1 >= 0:
            action_idx_map[n - m1] = len(left_action_nodes)
            left_action_nodes.append(n - m1)
        else:
            cast_idx_map[n] = len(left_cast_nodes)
            left_cast_nodes.append(n)

    new_vcc = []
    for i, j in vcc:
        if i in left_cast_nodes and j in left_cast_nodes:
            new_vcc.append([cast_idx_map[i], cast_idx_map[j]])

    new_vca = []
    for i, j in vca:
        if i in left_cast_nodes and j in left_action_nodes:
            new_vca.append([cast_idx_map[i], action_idx_map[j]])

    new_cast_score = cast_score[left_cast_nodes]
    new_action_score = action_score[left_action_nodes]

    # === gen init Y ===
    new_sln_c = [cast_idx_map[sln] if sln != -1 else -1 for sln in sln_cast]
    new_sln_a = [
        action_idx_map[sln] if sln != -1 else -1 for sln in sln_action
    ]

    Y1 = convert_mapping_to_mat(new_sln_c, new_cast_score.shape)
    Y2 = convert_mapping_to_mat(new_sln_a, new_action_score.shape)
    init_Y = block_diag(Y1, Y2).flatten().astype(np.float32)

    return (new_cast_score, new_action_score, left_cast_nodes,
            left_action_nodes, new_vcc, new_vca, scc, sca, init_Y)
