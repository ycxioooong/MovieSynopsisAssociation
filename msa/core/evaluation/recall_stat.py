from .recall import recall_2d

__all__ = ['calc_stat', 'get_stat', 'print_stat', 'log_stat']


def calc_stat(score, topk, title, s2c=True, c2s=True):
    stat = dict()

    if s2c:
        top_rc, med, mean = recall_2d(score=score.T, k=topk)
        ret_stat = get_stat(topk, top_rc, med, mean, title, 'S2C')
        stat.update(ret_stat)
    if c2s:
        top_rc, med, mean = recall_2d(score=score, k=topk)
        ret_stat = get_stat(topk, top_rc, med, mean, title, 'C2S')
        stat.update(ret_stat)
    return stat


def get_stat(topk, top_rc, med, mean, title, direction):
    stat = {
        '<{}|{}|Recall@{}>'.format(title, direction, k_): rc * 100
        for rc, k_ in zip(top_rc, topk)
    }
    stat['<{}|{}|Med>'.format(title, direction)] = med
    stat['<{}|{}|Mean>'.format(title, direction)] = mean
    return stat


def print_stat(stat):
    for key, val in stat.items():
        print('{}\t{}'.format(key, val))


def log_stat():
    pass