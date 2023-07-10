###################################################################################################################
# code reference https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/gather_excite.py
# Paper: `Gather-Excite: Exploiting Feature Context in CNNs` - https://arxiv.org/abs/1810.12348
###################################################################################################################

import torch
from torch import nn
from timm.models.layers import create_attn


class ge_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(ge_module, self).__init__()
        self.channel = create_attn(attn_type='ge', channels=in_channels)

    @staticmethod
    def get_module_name():
        return "ge"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    model = ge_module(64)
    y = model(x)
    print(y.shape)

