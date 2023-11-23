#############################################################################################################################################
# code reference https://github.com/AlexHex7/Non-local_pytorch/blob/master/lib/non_local_dot_product.py
# https://github.com/rwightman/pytorch-image-models/blob/f7325c7b712100f79a9ab4ae54118d259c11bacf/timm/models/layers/non_local_attn.py
# Paper: `Non-local Neural Networks` - https://arxiv.org/abs/1711.07971
#############################################################################################################################################


import torch
from torch import nn
from timm.models.layers import create_attn


class nonlocal_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(nonlocal_module, self).__init__()
        self.channel = create_attn(attn_type='nl', channels=in_channels)

    @staticmethod
    def get_module_name():
        return "nonlocal"

    def forward(self, x):
        y = self.channel(x)

        return y


if __name__ == '__main__':
    x = torch.rand(2, 3, 24, 24)
    model = nonlocal_module(3)
    y = model(x)
    print(y.shape)