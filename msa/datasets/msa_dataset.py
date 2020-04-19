import os
import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from ..utils import to_tensor, reduce_to_np


class MSADataset(Dataset):

    def __init__(self,
                 ann_file,
                 prefix,
                 element='appr',
                 test_mode=False,
                 indims=(512, 512)):
        self.ann_file = ann_file
        self.prefix = prefix
        self.element = element
        assert self.element in ['appr', 'action', 'cast', 'graph']
        self.test_mode = test_mode
        self.indims = indims

        # prepare ann file and proposals
        self.ann_info = self._load_annotations(ann_file)

    def __len__(self):
        return len(self.ann_info)

    def _load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_data(idx)
        else:
            while True:
                data = self.prepare_data(idx)
                if data is None:
                    idx = np.random.choice(len(self))
                    continue
                return data

    def _load_feat(self,
                   data,
                   element,
                   type_,
                   start=None,
                   end=None,
                   syn_id=None):
        data = data['{}.{}'.format(type_, element)]
        if element == 'appr' and type_ == 'video':
            d = data[start:end]
            return d
        elif element == 'appr' and type_ == 'synopsis':
            return data[syn_id]
        elif element == 'action' and type_ == 'video':
            actions = data[start:end]
            return reduce_to_np(actions)
        elif element == 'action' and type_ == 'synopsis':
            actions = data[syn_id]['action_w2v']
            return reduce_to_np(actions)
        elif element == 'cast' and type_ == 'video':
            shot2cast = data['shot2cast'][start:end]
            cid_start = shot2cast[0][0]
            cid_end = shot2cast[-1][1]
            return data['cast_features'][cid_start:cid_end]
        elif element == 'cast' and type_ == 'synopsis':
            syn_cast = data[syn_id]
            if syn_cast is None:
                return np.zeros((0, ))
            else:
                return np.array(syn_cast['feature'])
        else: 
            raise ValueError('Error element {} or type_ {}'.format(element, type_))

    def prepare_data(self, idx):
        info = self.ann_info[idx]
        mid = info['mid']
        bbox = info['bbox']
        syn_id = info['syn_id']
        data = mmcv.load(osp.join(self.prefix, '{}.pkl'.format(mid)))

        # get video feat
        clip = self._load_feat(
            data,
            self.element,
            'video',
            start=bbox[0],
            end=bbox[1] + 1,
            syn_id=syn_id)
        clip = clip.astype(np.float32)

        if clip.shape[0] == 0:
            if self.test_mode:
                clip = np.zeros((1, self.indims[0]), dtype=np.float32)
            else:
                return None

        #print(syn_id, type(syn_id))
        # get syn feat, syn bbox
        syn = self._load_feat(
            data,
            self.element,
            'synopsis',
            start=bbox[0],
            end=bbox[1] + 1,
            syn_id=syn_id)
        syn = syn.astype(np.float32)

        if syn.shape[0] == 0:
            if self.test_mode:
                syn = np.zeros((1, self.indims[1]), dtype=np.float32)
            else:
                return None

        meta = dict(bbox=bbox, mid=mid)
        meta['c_{}_len'.format(self.element)] = clip.shape[0]
        meta['s_{}_len'.format(self.element)] = syn.shape[0]

        data = dict(
            clip=DC(to_tensor(clip[None, ...]), stack=True),
            meta=DC(meta, cpu_only=True),
            syn=DC(to_tensor(syn[None, ...]), stack=True))

        return data
