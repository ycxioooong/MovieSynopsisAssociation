from collections import OrderedDict

import torch
import torch.nn.functional as F
from msa.core import get_score_efw, recall_2d


def bp_dtw(model,
           data,
           train_mode,
           loss=None,
           topk=(1, ),
           only_syn2clip=False,
           key='none'):

    log_vars = OrderedDict()

    outputs = model(**data)
    ce = outputs.pop('clip_{}_embed'.format(key))
    se = outputs.pop('syn_{}_embed'.format(key))
    cl = outputs.pop('clip_{}_len'.format(key))
    sl = outputs.pop('syn_{}_len'.format(key))

    # big_score = torch.mm(l2norm(ce), l2norm(se).t())
    big_score = torch.mm(F.normalize(ce), F.normalize(se).t())

    score = get_score_efw(big_score,
                          cl.cpu().numpy().tolist(),
                          sl.cpu().numpy().tolist(), ce, se)

    loss_clip2syn_efw, loss_syn2clip_efw = loss(score)
    if only_syn2clip:
        outputs.update(dict(loss_syn2clip_efw=loss_syn2clip_efw))
    else:
        outputs.update(
            dict(
                loss_clip2syn_efw=loss_clip2syn_efw,
                loss_syn2clip_efw=loss_syn2clip_efw))

    top_rc, _, _ = recall_2d(score=score.detach().cpu().numpy().T, k=topk)
    recall_syn2clip = {
        'recall_syn2clip_@{}'.format(k_): rc * 100
        for rc, k_ in zip(top_rc, topk)
    }
    outputs.update(recall_syn2clip)

    top_rc, _, _ = recall_2d(score=score.detach().cpu().numpy(), k=topk)
    recall_clip2syn = {
        'recall_clip2syn_@{}'.format(k_): rc * 100
        for rc, k_ in zip(top_rc, topk)
    }
    outputs.update(recall_clip2syn)

    for output_name, output_val in outputs.items():
        if isinstance(output_val, torch.Tensor):
            log_vars[output_name] = output_val
        elif isinstance(output_val, list):
            log_vars[output_name] = sum(_val.mean() for _val in output_val)
        else:
            log_vars[output_name] = output_val

    loss = sum(_value for _key, _value in log_vars.items() if 'loss' in _key)
    log_vars['loss'] = loss

    for name in log_vars:
        if isinstance(log_vars[name], torch.Tensor):
            log_vars[name] = log_vars[name].item()

    nbatch = len(data['meta'].data[0])
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=nbatch)

    return outputs
