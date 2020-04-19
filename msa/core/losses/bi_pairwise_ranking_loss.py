import torch
from torch import nn


class BiPairwiseRankingLoss(nn.Module):

    def __init__(self, margin=0, max_violation=False):

        super(BiPairwiseRankingLoss, self).__init__()
        self.max_violation = max_violation
        self.margin = margin

    def forward(self, matrix):

        nquery = matrix.size(0)
        gt = matrix.diag()

        # compute loss, mask and compute masked loss
        loss_row_as_query = (self.margin + matrix - gt[..., None]).clamp(min=0)
        loss_col_as_query = (self.margin + matrix - gt).clamp(min=0)
        diag_mask = matrix.new_tensor(torch.eye(nquery), dtype=torch.uint8)
        loss_row_as_query = loss_row_as_query.masked_fill_(diag_mask, 0)
        loss_col_as_query = loss_col_as_query.masked_fill_(diag_mask, 0)

        if self.max_violation:
            loss_row_as_query = loss_row_as_query.max(1)[0]
            loss_col_as_query = loss_row_as_query.max(0)[0]

        return loss_row_as_query.mean(), loss_col_as_query.mean()