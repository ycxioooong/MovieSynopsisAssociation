from collections import OrderedDict
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
if mp.get_start_method(allow_none=True) is None:
    mp.set_start_method('spawn')

from msa.core import (get_score_efw, get_score_gm_approx)


def bp_em(model,
          data,
          train_mode,
          loss=None,
          topk=(1, ),
          bidirection=True,
          joint_refine=True,
          wg=0.3):

    log_vars = OrderedDict()

    outputs = model(**data)
    ce_cast = outputs.pop('clip_cast_embed')
    se_cast = outputs.pop('syn_cast_embed')
    cl_cast = outputs.pop('clip_cast_len')
    sl_cast = outputs.pop('syn_cast_len')
    raw_score_cast = torch.mm(F.normalize(ce_cast), F.normalize(se_cast).t())

    ce_action = outputs.pop('clip_action_embed')
    se_action = outputs.pop('syn_action_embed')
    cl_action = outputs.pop('clip_action_len')
    sl_action = outputs.pop('syn_action_len')
    raw_score_action = torch.mm(
        F.normalize(ce_action),
        F.normalize(se_action).t())

    ce_appr = outputs.pop('clip_appr_embed')
    se_appr = outputs.pop('syn_appr_embed')
    cl_appr = outputs.pop('clip_appr_len')
    sl_appr = outputs.pop('syn_appr_len')

    metas = data['meta'].data[0]
    vg = [m['video_graph'] for m in metas]
    sg = [m['syn_graph'] for m in metas]

    score = get_score_gm_approx(raw_score_cast,
                                cl_cast.cpu().numpy().tolist(),
                                sl_cast.cpu().numpy().tolist(),
                                raw_score_action,
                                cl_action.cpu().numpy().tolist(),
                                sl_action.cpu().numpy().tolist(), vg, sg)

    if joint_refine:
        raw_score_appr = torch.mm(
            F.normalize(ce_appr),
            F.normalize(se_appr).t())
        score_appr = get_score_efw(
            raw_score_appr,
            cl_appr.cpu().numpy().tolist(),
            sl_appr.cpu().numpy().tolist(),
            ce_appr,
            se_appr,
            nproc=1)
        score = score + score_appr * wg

    loss_clip2syn_basic, loss_syn2clip_basic = loss(score)
    if not bidirection:
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
