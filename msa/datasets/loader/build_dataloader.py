from functools import partial

from mmcv.runner import get_dist_info
from mmcv.parallel import collate
from torch.utils.data import DataLoader

from .sampler import RandomSampler, DistributedRandomSampler

# https://github.com/pytorch/pytorch/issues/973
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def build_dataloader(dataset,
                     tasks_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     customized_sampler=False,
                     **kwargs):
    sampler = None
    if dist:
        rank, world_size = get_dist_info()
        if customized_sampler:
            sampler = DistributedRandomSampler(dataset, tasks_per_gpu,
                                               world_size, rank)
        batch_size = tasks_per_gpu
        num_workers = workers_per_gpu
    else:
        if not kwargs.get('shuffle', True):
            print('[NOTE] shuffle == False with no sampler taking effect.')
        else:
            if customized_sampler:
                print('Shuffle with customized sampler.')
                sampler = RandomSampler(dataset, tasks_per_gpu)
        batch_size = num_gpus * tasks_per_gpu
        num_workers = num_gpus * workers_per_gpu

    data_loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=tasks_per_gpu),
        pin_memory=False,
        **kwargs)

    return data_loader
