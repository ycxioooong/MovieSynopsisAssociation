import torch
from torch import nn
from mmcv.runner import load_checkpoint
import logging

from msa.utils import l2norm, to_numpy
from .. import builder


class CrossModalNet(nn.Module):

    def __init__(self,
                 video_embeder,
                 syn_embeder,
                 train_cfg=None,
                 test_cfg=None,
                 reduce_flag=True,
                 pretrained=None,
                 key='none'):
        super(CrossModalNet, self).__init__()
        self.vembeder = builder.build_embeder(video_embeder)
        self.sembeder = builder.build_embeder(syn_embeder)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.reduce_flag = reduce_flag
        self.init_weights(pretrained=pretrained)
        self.key = key

    def init_weights(self, pretrained=None):
        if pretrained is not None:
            logger = logging.getLogger()
            logger.info('load model from: {}'.format(pretrained))
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            self.vembeder.init_weights()
            self.sembeder.init_weights()

    def forward(self, syn, clip, meta, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(syn, clip, meta, **kwargs)
        else:
            return self.forward_train(syn, clip, meta, **kwargs)

    def forward_train(self, syn, clip, meta):
        """ syn B*Fs, clip B*N*Fc
        """

        ## fix dataparrallel bug
        clip = clip.squeeze(dim=1)
        syn = syn.squeeze(dim=1)
        outputs = dict()

        clens = [m['c_{}_len'.format(self.key)] for m in meta]
        slens = [m['s_{}_len'.format(self.key)] for m in meta]

        syn_embed, syn_len = self.sembeder(syn, slens)
        clip_embed, clip_len = self.vembeder(clip, clens)

        outputs = dict()
        outputs['syn_{}_embed'.format(self.key)] = syn_embed
        outputs['clip_{}_embed'.format(self.key)] = clip_embed

        if not self.reduce_flag:
            outputs['syn_{}_len'.format(self.key)] = syn_len
            outputs['clip_{}_len'.format(self.key)] = clip_len
        return outputs
