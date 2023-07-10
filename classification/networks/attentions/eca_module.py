########################################################################################################################
# code reference https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
# paper: ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks - https://arxiv.org/abs/1910.03151
########################################################################################################################

import torch
from torch import nn
from torch.nn.parameter import Parameter

class eca_module(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, in_channels, reduction=16, k_size=3):
        super(eca_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def get_module_name():
        return "eca"

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

# # For Avoiding global or local context modeling separately
# # ECA + att(gc) or att(eca+gc)
# class GL(nn.Module):
#     def __init__(self, inplanes, ratio=1/4., k_size=3, pooling_type='att', fusion_types='channel_add'):
#         super(GL, self).__init__()
#         self.inplanes = inplanes
#         self.ratio = ratio
#         self.planes = int(inplanes * ratio)

#         self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
#         self.softmax = nn.Softmax(dim=2)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

#         self.sigmoid = nn.Sigmoid()

#     def att_gc(self, x):
#         batch, channel, height, width = x.size()
#         input_x = x
#         # [N, C, H * W]
#         input_x = input_x.view(batch, channel, height * width)
#         # [N, 1, C, H * W]
#         input_x = input_x.unsqueeze(1)
#         # [N, 1, H, W]
#         context_mask = self.conv_mask(x)
#         # [N, 1, H * W]
#         context_mask = context_mask.view(batch, 1, height * width)
#         # [N, 1, H * W]
#         context_mask = self.softmax(context_mask)
#         # [N, 1, H * W, 1]
#         context_mask = context_mask.unsqueeze(-1)
#         # [N, 1, C, 1]
#         context_GC = torch.matmul(input_x, context_mask)
#         # [N, C, 1, 1]
#         context_GC = context_GC.view(batch, channel, 1, 1)

#         return context_GC

#     def att_eca(self, x):
#         context_ECA = self.avg_pool(x)
#         context_ECA = self.conv(context_ECA.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         return context_ECA


#     def forward(self, x):
#         # [N, C, 1, 1]
#         # context = 0.5 * self.att_gc(x) + 0.5 * self.att_eca(x)
#         context = self.att_gc(x)

#         context = self.sigmoid(context)

#         return x * context.expand_as(x)

# class eca_module(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super(eca_module, self).__init__()
#         self.channel = GL(inplanes=in_channels)

#     @staticmethod
#     def get_module_name():
#         return "eca"

#     def forward(self, x):
#         y = self.channel(x)

#         return y