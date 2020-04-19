import argparse
from datetime import datetime
import sys

from batch_processors import batch_processors
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import Runner, obj_from_dict
from msa import datasets
from msa.apis import get_root_logger, set_random_seed
from msa.datasets import build_dataloader
from msa.models import build_loss, build_model


def train_model(model,
                datasets,
                cfg,
                distributed=False,
                validate=False,
                logger=None,
                **kwargs):
    if logger is None:
        logger = get_root_logger(cfg.log_level)
    _non_dist_train(model, datasets, cfg, validate=validate, **kwargs)


def _non_dist_train(model, dataset_names, cfg, validate=False, **kwargs):

    # prepare data loaders
    data_loaders = [
        build_dataloader(
            dataset,
            cfg.data.tasks_per_gpu,
            cfg.data.workers_per_gpu,
            cfg.gpus,
            dist=False,
            customized_sampler=not dataset.test_mode)
        for dataset in dataset_names
    ]

    # put model on gpus
    model = MMDataParallel(model, device_ids=range(cfg.gpus)).cuda()

    # build runner
    runner = Runner(model, batch_processors[cfg.batch_processor_type],
                    cfg.optimizer, cfg.work_dir, cfg.log_level)
    runner.register_training_hooks(cfg.lr_config, cfg.optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config)

    # validate, if any
    if validate:
        hook_dataset = obj_from_dict(cfg.data.test, datasets)
        loader = build_dataloader(
            hook_dataset,
            cfg.data.tasks_per_gpu // cfg.data.tasks_per_gpu,
            max(1, cfg.data.workers_per_gpu // cfg.data.tasks_per_gpu),
            cfg.gpus,
            dist=False,
            customized_sampler=False,
            shuffle=False)
        runner.register_hook(
            getattr(sys.modules[__name__],
                    cfg.eval_hook['type'])(loader, **cfg.eval_hook['args']))

    # resume
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    runner.run(data_loaders, cfg.workflow, cfg.total_epochs, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    parser.add_argument(
        '--validate',
        action='store_true',
        help='whether to evaluate the checkpoint during training')
    parser.add_argument(
        '--gpus',
        type=int,
        default=1,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    # === config ===
    cfg = Config.fromfile(args.config)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    cfg.gpus = args.gpus
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(time=datetime.now(), config=cfg.text)

    # === logger, random seed ===
    logger = get_root_logger(cfg.log_level)
    if args.seed is not None:
        logger.info('Set random seed to {}'.format(args.seed))
        set_random_seed(args.seed)

    # === model, loss, dataset ===
    loss = build_loss(cfg.loss)
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    train_dataset = obj_from_dict(cfg.data.train, datasets)
    val_dataset = obj_from_dict(cfg.data.val, datasets)

    # === get started! ===
    train_model(
        model, [train_dataset, val_dataset],
        cfg,
        distributed=False,
        validate=args.validate,
        logger=logger,
        loss=loss,
        topk=cfg.topk,
        key=cfg.key,
        bidirection=cfg.bidirection)


if __name__ == '__main__':
    main()
