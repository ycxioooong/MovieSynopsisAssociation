import torch
from torch import nn
#from msa.utils import group_l2norm
import torch.nn.functional as F


class MLPEmbeder(nn.Module):

    def __init__(self,
                 indim,
                 hidden_dim=1024,
                 outdim=256,
                 with_bn=True,
                 norm_opt='none',
                 reduce_opt='none'):
        super(MLPEmbeder, self).__init__()
        self.fc1 = nn.Linear(indim, hidden_dim)
        if with_bn:
            self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, outdim)
        self.with_bn = with_bn
        self.norm_opt = norm_opt
        self.reduce_opt = reduce_opt
        # assert self.norm_opt in ['norm', 'gnorm', 'none']
        assert self.norm_opt in ['norm', 'none']

        assert self.reduce_opt in ['sum', 'mean', 'reduce_norm', 'none']

    def norm_reduce(self, inp, l):

        out = inp[:l]
        if self.norm_opt == 'norm':
            out = F.normalize(out)
        # else: do nothing
        # elif self.norm_opt == 'gnorm':
        #    out = group_l2norm(out)

        if self.reduce_opt == 'reduce_norm':
            out = F.normalize(out.mean(0, keepdim=True))
        elif self.reduce_opt == 'sum':
            out = out.sum(0, keepdim=True)
        elif self.reduce_opt == 'mean':
            out = out.mean(0, keepdim=True)
        return out

    def forward(self, x, lens):

        x = self.fc1(x)
        if self.with_bn:
            x = self.bn1(x)
        x = self.relu(x)
        embed = self.fc2(x)

        embed_lst = []
        length_lst = []
        for e, length in zip(embed, lens):
            out = self.norm_reduce(e, length)
            embed_lst.append(out)
            length_lst.append(length)
        seq_embed = torch.cat(embed_lst, 0)
        return seq_embed, seq_embed.new_tensor(length_lst, dtype=torch.long)

    def init_weights(self):

        nn.init.normal_(self.fc1.weight, 0, 0.01)
        nn.init.constant_(self.fc1.bias, 0)

        nn.init.normal_(self.fc2.weight, 0, 0.01)
        nn.init.constant_(self.fc2.bias, 0)

        if self.with_bn:
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
