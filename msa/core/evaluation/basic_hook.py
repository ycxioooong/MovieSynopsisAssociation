from mmcv.runner import Hook
import os.path as osp
import mmcv
import shutil
import torch


class JointEvalHook(Hook):

    def __init__(self, dataloader, interval=1):
        self.loader = dataloader
        self.interval = interval
        self.lock_dir = None

    def before_run(self, runner):
        self.lock_dir = osp.join(runner.work_dir, '.lock_map_hook')
        if osp.exists(self.lock_dir):
            shutil.rmtree(self.lock_dir)
        mmcv.mkdir_or_exist(self.lock_dir)

    def after_run(self, runner):
        shutil.rmtree(self.lock_dir)

    def after_train_epoch(self, runner):
        if not self.every_n_epochs(runner, self.interval):
            return
        runner.model.eval()
        results = []
        prog_bar = mmcv.ProgressBar(len(self.loader))
        for data in self.loader:
            # compute output
            with torch.no_grad():
                result = runner.model(return_loss=False, **data)
            results.append(result)
            prog_bar.update()
        self.evaluate(runner, results)

    def evaluate(self, runner, results):
        raise NotImplementedError


class DistJointEvalHook(Hook):

    def __init__(self):
        raise NotImplementedError()