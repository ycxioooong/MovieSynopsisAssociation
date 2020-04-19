import torch
from torch import nn
import torch.nn.functional as F


class FCEmbeder(nn.Module):

    def __init__(self, indim, norm_opt='none', reduce_opt='none'):
        super(FCEmbeder, self).__init__()
        self.fc = nn.Linear(indim, indim)
        self.norm_opt = norm_opt
        self.reduce_opt = reduce_opt
        assert self.norm_opt in ['norm', 'none']

        assert self.reduce_opt in ['sum', 'mean', 'reduce_norm', 'none']
        self.init_weights()

    def norm_reduce(self, inp, l):

        out = inp[:l]
        if self.norm_opt == 'norm':
            out = F.normalize(out)
        # elif self.norm_opt == 'gnorm':
        #     out = group_l2norm(out)

        if self.reduce_opt == 'reduce_norm':
            out = F.normalize(out.mean(0, keepdim=True))
        elif self.reduce_opt == 'sum':
            out = out.sum(0, keepdim=True)
        elif self.reduce_opt == 'mean':
            out = out.mean(0, keepdim=True)
        return out

    def forward(self, x, lens):

        embed = self.fc(x)
        embed_lst = []
        length_lst = []
        for e, length in zip(embed, lens):
            out = self.norm_reduce(e, length)
            embed_lst.append(out)
            length_lst.append(length)
        seq_embed = torch.cat(embed_lst, 0)
        return seq_embed, seq_embed.new_tensor(length_lst, dtype=torch.long)

    def init_weights(self):

        nn.init.eye_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
