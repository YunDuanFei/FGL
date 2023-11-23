################################################################################################################
# code reference https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/split_attn.py
# Paper: `ResNeSt: Split-Attention Networks` - /https://arxiv.org/abs/2004.08955
################################################################################################################

import torch
from torch import nn
from timm.models.layers import create_attn


class splat_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(splat_module, self).__init__()
        self.channel = create_attn(attn_type='splat', channels=in_channels)

    @staticmethod
    def get_module_name():
        return "splat"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    model = splat_module(64)
    y = model(x)
    print(y.shape)

