#######################################################################################################################
## code reference https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/selective_kernel.py
#######################################################################################################################

import torch
from torch import nn
from timm.models.layers import create_attn


class sk_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(sk_module, self).__init__()
        self.channel = create_attn(attn_type='sk', channels=in_channels)

    @staticmethod
    def get_module_name():
        return "sk"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    model = sk_module(64)
    y = model(x)
    print(y.shape)

