from collections import Sequence

import numpy as np

import mmcv
import torch

__all__ = ['to_numpy', 'to_tensor', 'tensor_np_repeat']


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.
    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(
            type(data)))


def to_numpy(tensor):
    """ GPU tensor to numpy
    """
    return tensor.detach().cpu().numpy()


def tensor_np_repeat(tensor, times, dim=-1):
    if isinstance(times, torch.Tensor):
        times = times.cpu().numpy()
    x = tensor.cpu().numpy()
    np.repeat(x, times, axis=dim)
    return tensor.new_tensor(x)
