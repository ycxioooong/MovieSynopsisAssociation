import os.path as osp
import mmcv


class GraphDataset(object):

    def __init__(self, ann_file, prefix, test_mode=False):
        self.ann_file = ann_file
        self.prefix = prefix
        self.test_mode = test_mode

        # prepare ann file and proposals
        self.ann_info = self._load_annotations(ann_file)

    def __len__(self):
        return len(self.ann_info)

    def _load_annotations(self, ann_file):
        return mmcv.load(ann_file)

    def __getitem__(self, idx):
        return self.prepare_pair(idx)

    def _load_graph(self, mid, syn_id):
        data = mmcv.load(osp.join(self.prefix, '{}.pkl'.format(mid)))
        return data['video.graph'], data['synopsis.graph'][syn_id]

    def prepare_pair(self, idx):
        info = self.ann_info[idx]
        mid = info['mid']
        bbox = info['bbox']
        b0, b1 = bbox
        b1 = b1 + 1
        syn_id = info['syn_id']

        # get video feat
        video, syn = self._load_graph(mid, syn_id)

        vcc = video['cast_cast'][b0:b1]
        vct = video.get('cast_temporal', None)
        if vct is not None:
            vct = vct[b0:b1 - 1]
        vco = video['offset_cast'][b0:b1]
        vca = video['cast_action'][b0:b1]
        vao = video['offset_action'][b0:b1]

        scc = syn['cast_cast_graph']
        sco = syn['offset_cast']
        sca = syn['cast_action_graph']
        sao = syn['offset_action']

        meta = dict(
            vct=vct,
            vcc=vcc,
            vco=vco,
            vca=vca,
            vao=vao,
            bbox=bbox,
            mid=mid,
            syn_id=syn_id,
            scc=scc,
            sco=sco,
            sca=sca,
            sao=sao)

        return meta
