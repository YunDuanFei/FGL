########################################################################################################
# https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/squeeze_excite.py
# Paper: `CenterMask : Real-Time Anchor-Free Instance Segmentation` - https://arxiv.org/abs/1911.06667
########################################################################################################

import torch
from torch import nn
from timm.models.layers import create_attn


class ese_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ese_module, self).__init__()
        self.channel = create_attn(attn_type='ese', channels=in_channels)

    @staticmethod
    def get_module_name():
        return "ese"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    model = ese_module(64)
    y = model(x)
    print(y.shape)