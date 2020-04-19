import numpy as np
import mmcv
import torch


def pad(img, shape, pad_val=0):
    """Pad an image to a certain shape.
    Args:
        img (ndarray): Image to be padded.
        shape (tuple): Expected padding shape.
        pad_val (number or sequence): Values to be filled in padding areas.
    Returns:
        ndarray: The padded image.
    """
    if not isinstance(pad_val, (int, float)):
        assert len(pad_val) == img.shape[-1]
    if len(shape) < len(img.shape):
        shape = shape + (img.shape[-1], )
    assert len(shape) == len(img.shape)
    for i in range(len(shape) - 1):
        assert shape[i] >= img.shape[i]
    pad = np.empty(shape, dtype=img.dtype)
    pad[...] = pad_val
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad


def pad_to_multiple(arr, divisor, axis=None, pad_val=0):
    """Pad to size
    """
    shapes = list(arr.shape)
    for ax in axis:
        shapes[ax] = int(np.ceil(shapes[ax] / divisor)) * divisor
    return pad(arr, tuple(shapes), pad_val)


class ImageTransform(object):
    """Preprocess an image.
    1. rescale the image to expected size
    2. normalize the image
    3. flip the image (if needed)
    4. pad the image (if needed)
    5. transpose to (c, h, w)
    """

    def __init__(self, mean=(0, 0, 0), std=(1, 1, 1), to_rgb=True, size=None):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.size = size

    def __call__(self, img):
        if self.size is not None:
            img = mmcv.imresize(img, self.size)
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        img = img.transpose(2, 0, 1)
        return img


class VideoTransform(object):
    """ 
    1. trans video feature to 3d.
    2. pad video in order to fit rpn down sampling.
    3. transpose to (c, h, w)
    """

    def __init__(self, size_divisor):
        self.size_divisor = size_divisor

    def __call__(self, video):
        video = video[None, ...]
        video_shape = video.shape
        video = pad_to_multiple(video, self.size_divisor, axis=(1, ))
        pad_shape = video.shape
        video = video.transpose(2, 0, 1)
        return video, video_shape, pad_shape


class BboxTransform(object):
    """Preprocess gt bboxes.
    1. pad the first dimension to `max_num_gts`
    """

    def __init__(self, max_num_gts=None):
        self.max_num_gts = max_num_gts

    def __call__(self, bboxes, img_shape, flip=False):

        if self.max_num_gts is None:
            return bboxes
        else:
            num_gts = bboxes.shape[0]
            padded_bboxes = np.zeros((self.max_num_gts, 2), dtype=np.float32)
            padded_bboxes[:num_gts, :] = bboxes
            return padded_bboxes


class Numpy2Tensor(object):

    def __init__(self):
        pass

    def __call__(self, *args):
        if len(args) == 1:
            return torch.from_numpy(args[0])
        else:
            return tuple([torch.from_numpy(np.array(array)) for array in args])