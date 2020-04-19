import torch
from torch import nn
from mmcv.runner import load_checkpoint
import logging

from msa.utils import l2norm, to_numpy
from .. import builder


class Identity(nn.Module):

    def __init__(self, train_cfg=None, test_cfg=None, pretrained=None):
        super(Identity, self).__init__()

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        pass

    def forward(self, syn, clip, meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(syn, clip, meta, **kwargs)
        else:
            return self.forward_train(syn, clip, meta, **kwargs)

    def forward_train(self, syn, clip, meta, key='cast'):
        """ syn B*Fs, clip B*N*Fc
        """

        ## fix dataparrallel bug
        clip = clip.squeeze(dim=1)
        syn = syn.squeeze(dim=1)
        outputs = dict()

        clens = [m['c_{}_len'.format(key)] for m in meta]
        clip_embed = []
        for c, l in zip(clip, clens):
            clip_embed.append(c[:l])
        clip_embed = torch.cat(clip_embed, 0)
        clip_len = clip_embed.new_tensor(clens, dtype=torch.long)

        slens = [m['s_{}_len'.format(key)] for m in meta]
        syn_embed = []
        for s, l in zip(syn, slens):
            syn_embed.append(s[:l])
        syn_embed = torch.cat(syn_embed, 0)
        syn_len = syn_embed.new_tensor(slens, dtype=torch.long)

        outputs['syn_{}_embed'.format(key)] = syn_embed
        outputs['clip_{}_embed'.format(key)] = clip_embed
        outputs['syn_{}_len'.format(key)] = syn_len
        outputs['clip_{}_len'.format(key)] = clip_len

        return outputs
