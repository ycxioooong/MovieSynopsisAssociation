import numpy as np
from mmcv.runner import Hook


class LossWeightHook(Hook):

    def __init__(self, init_loss_weight, final_loss_weight, total_epoch):
        print(init_loss_weight, final_loss_weight, total_epoch)
        self.init_lw = init_loss_weight
        self.final_lw = final_loss_weight
        self.total_epoch = total_epoch
        self.all_loss_weights = self._interp_loss_weight()
        assert len(self.all_loss_weights) == total_epoch

    def _interp_loss_weight(self):

        all_lw = []
        for start, end in zip(self.init_lw, self.final_lw):
            if start == end:
                all_lw.append([start] * self.total_epoch)
            else:
                lw = np.arange(start, end,
                               (end - start) / float(self.total_epoch - 1))
                lw = lw.tolist() + [end]
                all_lw.append(lw)
        return list(zip(*all_lw))

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        runner.model.loss_weights = self.all_loss_weights[epoch]

    def before_val_epoch(self, runner):
        epoch = runner.epoch
        runner.model.loss_weights = self.all_loss_weights[epoch]
