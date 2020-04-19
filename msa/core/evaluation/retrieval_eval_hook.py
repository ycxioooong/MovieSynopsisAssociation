import mmcv
from .recall import recall_2d
from .basic_hook import JointEvalHook
from msa.utils import (accumulate_by_key, clamp_cdist, get_score_basic,
                       get_score_efw)

__all__ = ['RetrievalEvalHook', 'RetrievalOptimEvalHook']


class RetrievalEvalHook(JointEvalHook):

    def __init__(self, dataloader, interval=1, k=(1, 5, 10), key='ele'):
        super(RetrievalEvalHook, self).__init__(dataloader, interval)
        self.k = k
        self.key = key

    def evaluate(self, runner, results):

        clips = accumulate_by_key(results, 'clip_{}_embed'.format(self.key))
        syns = accumulate_by_key(results, 'syn_{}_embed'.format(self.key))
        top_rc, med, mean = recall_2d(syns, clips, k=self.k)
        stats = dict()

        recall_syn2clip = {
            'recall_syn2clip_@{}'.format(k_): rc * 100
            for rc, k_ in zip(top_rc, self.k)
        }
        other = {'med_syn2clip': med, 'mean_syn2clip': mean}
        stats.update(recall_syn2clip)
        stats.update(other)

        top_rc, med, mean = recall_2d(clips, syns, k=self.k)
        recall_clip2syn = {
            'recall_clip2syn_@{}'.format(k_): rc * 100
            for rc, k_ in zip(top_rc, self.k)
        }
        other = {'med_clip2syn': med, 'mean_clip2syn': mean}
        stats.update(recall_clip2syn)
        stats.update(other)

        for key, val in stats.items():
            runner.log_buffer.output[key] = val
        runner.log_buffer.ready = True


class RetrievalOptimEvalHook(JointEvalHook):

    def __init__(self, dataloader, interval=1, k=(1, 5, 10), key='none'):
        super(RetrievalOptimEvalHook, self).__init__(dataloader, interval)
        self.k = k
        self.key = key

    def evaluate(self, runner, results):

        clips = accumulate_by_key(results, 'clip_{}_embed'.format(self.key))
        syns = accumulate_by_key(results, 'syn_{}_embed'.format(self.key))
        clip_len = accumulate_by_key(results, 'clip_{}_len'.format(self.key))
        syn_len = accumulate_by_key(results, 'syn_{}_len'.format(self.key))
        # seq_score = 1 - cdist(clips, syns, 'cosine')
        seq_score = clamp_cdist(clips, syns)

        ## Eval
        score_1 = get_score_basic(clips, syns, clip_len.tolist(),
                                  syn_len.tolist())

        top_rc, med, mean = recall_2d(score=score_1.T, k=self.k)
        stats = dict()

        recall_syn2clip = {
            '[Basic]recall_syn2clip_@{}'.format(k_): rc * 100
            for rc, k_ in zip(top_rc, self.k)
        }
        other = {'[Basic]med_syn2clip': med, '[Basic]mean_syn2clip': mean}
        stats.update(recall_syn2clip)
        stats.update(other)

        top_rc, med, mean = recall_2d(score=score_1, k=self.k)
        recall_clip2syn = {
            '[Basic]recall_clip2syn_@{}'.format(k_): rc * 100
            for rc, k_ in zip(top_rc, self.k)
        }
        other = {'[Basic]med_clip2syn': med, '[Basic]mean_clip2syn': mean}
        stats.update(recall_clip2syn)
        stats.update(other)

        for key, val in stats.items():
            runner.log_buffer.output[key] = val
        runner.log_buffer.ready = True

        ## Eval optim
        with mmcv.Timer():
            score_2, _, _ = get_score_efw(seq_score, clips, syns,
                                          clip_len.tolist(), syn_len.tolist())

        top_rc, med, mean = recall_2d(score=score_2.T, k=self.k)
        stats = dict()

        recall_syn2clip = {
            '[Optim]recall_syn2clip_@{}'.format(k_): rc * 100
            for rc, k_ in zip(top_rc, self.k)
        }
        other = {'[Optim]med_syn2clip': med, '[Optim]mean_syn2clip': mean}
        stats.update(recall_syn2clip)
        stats.update(other)

        top_rc, med, mean = recall_2d(score=score_2, k=self.k)
        recall_clip2syn = {
            '[Optim]recall_clip2syn_@{}'.format(k_): rc * 100
            for rc, k_ in zip(top_rc, self.k)
        }
        other = {'[Optim]med_clip2syn': med, '[Optim]mean_clip2syn': mean}
        stats.update(recall_clip2syn)
        stats.update(other)

        for key, val in stats.items():
            runner.log_buffer.output[key] = val
        runner.log_buffer.ready = True
