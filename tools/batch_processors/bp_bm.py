from collections import OrderedDict

import torch
import torch.nn.functional as F
from msa.core import get_score_bm, get_score_cosine_similarity


def bp_bm(model, data, train_mode, loss=None, topk=(1, ), only_syn2clip=False):

    log_vars = OrderedDict()

    outputs = model(**data)
    ce = outputs.pop('clip_ele_embed')
    se = outputs.pop('syn_ele_embed')
    cl = outputs.pop('clip_ele_len')
    sl = outputs.pop('syn_ele_len')

    raw_score = torch.mm(F.normalize(ce), F.normalize(se).t())

    score_bm = get_score_bm(raw_score, ce, se,
                            cl.cpu().numpy().tolist(),
                            sl.cpu().numpy().tolist())
    loss_clip2syn_bm, loss_syn2clip_bm = loss(score_bm)
    if only_syn2clip:
        outputs.update(dict(loss_syn2clip_bm=loss_syn2clip_bm))
    else:
        outputs.update(
            dict(
                loss_clip2syn_bm=loss_clip2syn_bm,
                loss_syn2clip_bm=loss_syn2clip_bm))

    score_basic = get_score_cosine_similarity(ce, se,
                                              cl.cpu().numpy().tolist(),
                                              sl.cpu().numpy().tolist())
    loss_clip2syn_basic, loss_syn2clip_basic = loss(score_basic)
    if only_syn2clip:
        outputs.update(dict(loss_syn2clip_basic=loss_syn2clip_basic))
    else:
        outputs.update(
            dict(
                loss_clip2syn_basic=loss_clip2syn_basic,
                loss_syn2clip_basic=loss_syn2clip_basic))

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

    nbatch = len(data['meta'].data)
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=nbatch)

    return outputs
