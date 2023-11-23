########################################################################################################################
# code reference https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/eca.py
# paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks - https://arxiv.org/abs/1910.03151
########################################################################################################################

import torch
from torch import nn
from timm.models.layers import create_attn


class eca_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(eca_module, self).__init__()
        self.channel = create_attn(attn_type='eca', channels=in_channels)

    @staticmethod
    def get_module_name():
        return "eca"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    model = eca_module(64)
    y = model(x)
    print(y.shape)