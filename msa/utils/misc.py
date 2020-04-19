import numpy as np


def reduce_to_np(lst):
    rst = []
    for item in lst:
        if isinstance(item, list):
            if item != []:
                rst += item
        else:
            if item is not None:
                rst.append(item)
    return np.array(rst)


def accumulate_by_key(lst_of_dct, key):
    if len(lst_of_dct) == 0 or lst_of_dct[0].get(key, None) is None:
        return None
    data = [item[key].cpu().numpy() for item in lst_of_dct]
    data = np.concatenate(data, axis=0)
    return data


def acc_list(x):
    y = [0]
    for x_ in x:
        y.append(y[-1] + x_)
    return y[1:]
