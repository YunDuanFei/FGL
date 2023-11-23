#########################################################################################################################################
# code reference https://github.com/implus/PytorchInsight/blob/master/classification/models/imagenet/resnet_sge.py
# Paper: `Spatial Group-wise Enhance: Enhancing Semantic Feature Learning in Convolutional Networks ` - https://arxiv.org/pdf/1905.09646
#########################################################################################################################################

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SpatialGroupEnhance(nn.Module):
    def __init__(self, groups = 64):
        super(SpatialGroupEnhance, self).__init__()
        self.groups   = groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight   = Parameter(torch.zeros(1, groups, 1, 1))
        self.bias     = Parameter(torch.ones(1, groups, 1, 1))
        self.sig      = nn.Sigmoid()

    def forward(self, x): # (b, c, h, w)
        b, c, h, w = x.size()
        x = x.view(b * self.groups, -1, h, w)
        xn = x * self.avg_pool(x)
        xn = xn.sum(dim=1, keepdim=True)
        t = xn.view(b * self.groups, -1)
        t = t - t.mean(dim=1, keepdim=True)
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std
        t = t.view(b, self.groups, h, w)
        t = t * self.weight + self.bias
        t = t.view(b * self.groups, 1, h, w)
        x = x * self.sig(t)
        x = x.view(b, c, h, w)
        return x

class sge_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(sge_module, self).__init__()
        self.channel = SpatialGroupEnhance()

    @staticmethod
    def get_module_name():
        return "sge"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    model = sge_module(64)
    y = model(x)
    print(y.shape)

