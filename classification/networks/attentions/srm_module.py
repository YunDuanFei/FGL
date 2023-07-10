################################################################################################################################
# code reference https://github.com/EvgenyKashin/SRMnet/blob/master/models/layer_blocks.py
# Paper: `SRM : A Style-based Recalibration Module for Convolutional Neural Networks` - https://arxiv.org/pdf/1903.10829v1.pdf
################################################################################################################################

import torch
from torch import nn


class SRMLayer(nn.Module):
    def __init__(self, channel, reduction=None):
        # Reduction for compatibility with layer_block interface
        super(SRMLayer, self).__init__()

        # CFC: channel-wise fully connected layer
        self.cfc = nn.Conv1d(channel, channel, kernel_size=2, bias=False,
                             groups=channel)
        self.bn = nn.BatchNorm1d(channel)

    def forward(self, x):
        b, c, _, _ = x.size()

        # Style pooling
        mean = x.view(b, c, -1).mean(-1).unsqueeze(-1)
        std = x.view(b, c, -1).std(-1).unsqueeze(-1)
        u = torch.cat((mean, std), -1)  # (b, c, 2)

        # Style integration
        z = self.cfc(u)  # (b, c, 1)
        z = self.bn(z)
        g = torch.sigmoid(z)
        g = g.view(b, c, 1, 1)

        return x * g.expand_as(x)



class srm_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(srm_module, self).__init__()
        self.channel = SRMLayer(channel=in_channels, reduction=reduction)

    @staticmethod
    def get_module_name():
        return "srm"

    def forward(self, x):
        y = self.channel(x)

        return y