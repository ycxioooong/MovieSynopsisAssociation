import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset

from ..utils import (to_tensor, reduce_syn_cast_action, reduce_syn_cast_cast,
                     reduce_video, reduce_to_np)


class EMDataset(Dataset):

    def __init__(self, ann_file, prefix, cast_unique=False, test_mode=False):
        self.ann_file = ann_file
        self.prefix = prefix
        self.test_mode = test_mode
        self.elements = ['appr', 'action', 'cast']
        self.indims = dict(action=(512, 300), cast=(512, 512))
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
        data = data['{}.{}'.format(element, type_)]
        if element == 'appr' and type_ == 'video':
            return data[start:end]
        elif element == 'appr' and type_ == 'syn':
            return data[syn_id]
        elif element == 'action' and type_ == 'video':
            actions = data[start:end]
            return reduce_to_np(actions)
        elif element == 'action' and type_ == 'syn':
            actions = data[syn_id]['action_w2v']
            return reduce_to_np(actions)
        elif element == 'cast' and type_ == 'video':
            shot2cast = data['shot2cast'][start:end]
            cid_start = shot2cast[0][0]
            cid_end = shot2cast[-1][1]
            return data['cast_features'][cid_start:cid_end]
        else:
            syn_cast = data[syn_id]
            if syn_cast is None:
                return np.zeros((0, ))
            else:
                return np.array(syn_cast['feature'])

    def _load_graph(self, type_, mid):
        data = mmcv.load(
            osp.join(self.prefix['graph'][type_], '{}.pkl'.format(mid)))
        return data

    def prepare_data(self, idx):
        data = dict()
        meta = dict()
        for ele in self.elements:
            d, m = self.prepare_pair(ele, idx)
            data.update(d)
            meta[ele] = m
        vg, sg = self.prepare_graph(idx)
        meta['video_graph'] = vg
        meta['syn_graph'] = sg
        data['meta'] = DC(meta, cpu_only=True)
        return data

    def prepare_graph(self, idx):
        info = self.ann_info[idx]
        mid = info['mid']
        bbox = info['bbox']
        b0, b1 = bbox
        b1 = b1 + 1
        syn_id = info['syn_id']

        # get video feat
        video = self._load_graph('video', mid)
        syn = self._load_graph('syn', mid)[syn_id]

        vcc = video['cast_cast'][b0:b1]
        vct = video.get('cast_temporal', None)
        if vct is not None:
            vct = vct[b0:b1 - 1]
            vcc += vct
        vco = video['offset_cast'][b0:b1]
        vca = video['cast_action'][b0:b1]
        vao = video['offset_action'][b0:b1]

        scc = syn['cast_cast_graph']
        sco = syn['offset_cast']
        sca = syn['cast_action_graph']
        sao = syn['offset_action']

        vcc = reduce_video(vcc, vco[0], vco[0])
        vca = reduce_video(vca, vco[0], vao[0])
        scc = reduce_syn_cast_cast(scc, sco)
        sca = reduce_syn_cast_action(sca, sco, sao)
        video_graph = dict(video_cast_cast=vcc, video_cast_action=vca)
        syn_graph = dict(syn_cast_cast=scc, syn_cast_action=sca)

        return video_graph, syn_graph

    def prepare_pair(self, element, idx):
        info = self.ann_info[idx]
        mid = info['mid']
        bbox = info['bbox']
        syn_id = info['syn_id']

        data = mmcv.load(osp.join(self.prefix, '{}.pkl'.format(mid)))
        clip = self._load_feat(
            data,
            element,
            'video',
            start=bbox[0],
            end=bbox[1] + 1,
            syn_id=syn_id)
        clip = clip.astype(np.float32)
        if clip.shape[0] == 0:
            clip = np.zeros((1, self.indims[element][0]), dtype=np.float32)

        # get syn feat, syn bbox
        syn = self._load_feat(data, element, 'synopsis', syn_id=syn_id)
        syn = syn.astype(np.float32)
        if syn.shape[0] == 0:
            syn = np.zeros((1, self.indims[element][1]), dtype=np.float32)

        meta = dict(syn_id=syn_id, bbox=bbox, mid=mid)
        meta['s_{}_len'.format(element)] = syn.shape[0]
        meta['c_{}_len'.format(element)] = clip.shape[0]

        data = dict()

        clip = DC(to_tensor(clip[None, ...]), stack=True)
        syn = DC(to_tensor(syn[None, ...]), stack=True)
        data['clip_{}'.format(element)] = clip
        data['syn_{}'.format(element)] = syn

        return data, meta
