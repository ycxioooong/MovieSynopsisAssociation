import argparse

import mmcv
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import (load_checkpoint, obj_from_dict)
from msa import datasets
from msa.core import (calc_stat, print_stat, get_score_cosine_similarity,
                      get_score_efw, get_score_bm)
from msa.datasets import build_dataloader
from msa.models import build_model
from msa.utils import accumulate_by_key, clamp_cdist


def evaluate(results, evals, topk=(1, 5, 10)):
    clips = accumulate_by_key(results, 'clip_ele_embed')
    syns = accumulate_by_key(results, 'syn_ele_embed')
    clip_len = accumulate_by_key(results, 'clip_ele_len')
    syn_len = accumulate_by_key(results, 'syn_ele_len')

    if 'basic' in evals:
        if clip_len is None:
            score_basic = clamp_cdist(clips, syns)
        else:
            score_basic = get_score_cosine_similarity(clips, syns,
                                                      clip_len.tolist(),
                                                      syn_len.tolist())
        stat = calc_stat(score_basic, topk, 'Basic')
        print_stat(stat)

    ## Eval optim
    for eval_method in evals:
        if eval_method == 'basic':
            continue
        with mmcv.Timer('Evaluating {}...'.format(eval_method)):
            seq_score = clamp_cdist(clips, syns)
            if eval_method == 'efm':
                title = 'EFW'
                score_optim, _, _ = get_score_efw(seq_score, clips, syns,
                                                  clip_len.tolist(),
                                                  syn_len.tolist())
            elif eval_method == 'bm':
                title = 'BM'
                score_optim, _, _ = get_score_bm(seq_score, clips, syns,
                                                 clip_len.tolist(),
                                                 syn_len.tolist())
            stat = calc_stat(score_optim, topk, title)
            print_stat(stat)


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
    parser.add_argument(
        '--eval',
        help='eval method',
        type=str,
        nargs='+',
        choices=['basic', 'efm', 'bm'])
    parser.add_argument('--topk', help='topk', type=int, nargs='+')
    args = parser.parse_args()

    return args


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

    topk = (1, 3, 5, 10) if not args.topk else args.topk
    evaluate(results, eval=args.eval, topk=topk)


if __name__ == '__main__':
    main()
