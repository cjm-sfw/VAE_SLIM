import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

class WeightedL1Loss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(WeightedL1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, weight=None, mask=None):
        if weight is not None:
            loss = torch.abs(input - target)
            loss = loss * weight
            loss = loss[mask]
            return torch.mean(loss)
        else:
            return torch.mean(torch.abs(input - target))