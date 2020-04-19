from torch import nn
from mmcv.runner import obj_from_dict

from . import modules
from . import embeddings
from msa.core import losses

__all__ = ['build_model', 'build_loss', 'build_embeder']


def _build_module(cfg, parrent=None, default_args=None):
    return cfg if isinstance(cfg, nn.Module) else obj_from_dict(
        cfg, parrent, default_args)


def build(cfg, parrent=None, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, parrent, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, parrent, default_args)


def build_loss(cfg):
    return build(cfg, losses)


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build(cfg, modules, dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_embeder(cfg):
    return build(cfg, embeddings)
