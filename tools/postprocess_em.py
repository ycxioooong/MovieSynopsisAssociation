import argparse
import os

import logging
import torch
import mmcv
from mmcv import Config
from mmcv.runner import (load_checkpoint, parallel_test, obj_from_dict, Runner,
                         DistSamplerSeedHook)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

# from torch.nn import DataParallel
import pytools
from datetime import datetime
from collections import OrderedDict
from scipy.spatial.distance import cdist

from torch import nn
import numpy as np
from batch_processors import batch_processors
from eval_tools import get_score_naive, get_score_dtw, get_score_bimatch
from utils import (acc_list, _accumulate_by_key, clamp_cdist, print_result,
                    gen_update_needed_mask, get_movie_mask)
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))
from graph_match import datasets
from msa.models import build_model, build_loss
from msa.datasets import build_dataloader
from msa.apis import (init_dist, set_random_seed, get_root_logger)
from msa.utils import (DistOptimizerHook)
from msa.core import (recall_2d, RetrievalOptimEvalHook,
                              RetrievalEvalHook, LossWeightHook)
from msa.ops import dtw
import multiprocessing as mp
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    args = parser.parse_args()

    return args


def gather_results(results):
    rst = dict(glob=[], action=[], cast=[])
    elements = ['glob', 'action', 'cast']
    for r in results:
        for ele in elements:
            dct = dict()
            dct['syn_ele_embed'] = r['syn_{}_embed'.format(ele)]
            dct['clip_ele_embed'] = r['clip_{}_embed'.format(ele)]
            dct['syn_ele_len'] = r['syn_{}_len'.format(ele)]
            dct['clip_ele_len'] = r['clip_{}_len'.format(ele)]
            rst[ele].append(dct)
    return rst['glob'], rst['action'], rst['cast']


def main():

    args = parse_args()

    # === config ===
    cfg = Config.fromfile(args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    loader = build_dataloader(
        dataset,
        cfg.data.tasks_per_gpu // cfg.data.tasks_per_gpu,
        max(1, cfg.data.workers_per_gpu // cfg.data.tasks_per_gpu),
        args.gpus,
        dist=False,
        customized_sampler=False,
        shuffle=False)
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model = MMDataParallel(model, device_ids=range(args.gpus)).cuda()
    load_checkpoint(model, args.checkpoint)
    model.eval()

    results = []
    prog_bar = mmcv.ProgressBar(len(loader))
    for data in loader:
        with torch.no_grad():
            result = model(return_loss=False, **data)
        results.append(result)
        prog_bar.update()

    results_glob, results_action, results_cast = gather_results(results)
    from IPython import embed
    embed()

def evaluate(results, method, title, topk=(1, 5, 10)):
    clips = _accumulate_by_key(results, 'clip_ele_embed')
    syns = _accumulate_by_key(results, 'syn_ele_embed')
    clip_len = _accumulate_by_key(results, 'clip_ele_len')
    syn_len = _accumulate_by_key(results, 'syn_ele_len')
    seq_score = clamp_cdist(clips, syns)

    ## Eval naive
    score_1 = get_score_naive(clips, syns, clip_len.tolist(), syn_len.tolist())
    print_result(score_1, title + '(naive)', topk)

    if method == 'dtw':
        func = get_score_dtw
    elif method == 'bimatch':
        func = get_score_bimatch
    else:
        return score_1
    ## Eval optim
    with mmcv.Timer():

        score_2, score_2_normed, ind_dict = func(seq_score, clips, syns,
                                                 clip_len.tolist(),
                                                 syn_len.tolist())

    print_result(score_2, title + '({})'.format(method), topk)

    score = score_1 * score_2
    print_result(score, title + '(ensemble)', topk)

    return score_1, score_2, score_2_normed, ind_dict


if __name__ == '__main__':
    main()