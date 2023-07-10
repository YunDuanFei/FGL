############################################################################################################################################
# code reference https://github.com/xvjiarui/GCNet/blob/master/mmdet/ops/gcb/context_block.py
# https://github.com/rwightman/pytorch-image-models/blob/f7325c7b712100f79a9ab4ae54118d259c11bacf/timm/models/layers/global_context.py#L19
# Paper: `GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond` - https://arxiv.org/abs/1904.11492
############################################################################################################################################

import torch
from torch import nn
from timm.models.layers import create_attn


class gc_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(gc_module, self).__init__()
        self.channel = create_attn(attn_type='gc', channels=in_channels)

    @staticmethod
    def get_module_name():
        return "gc"

    def forward(self, x):
        y = self.channel(x)

        return y

# # For Avoiding global or local context modeling separately
# # GC + att(eca) or att(eca+gc)
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

#         self.channel_add_conv = nn.Sequential(
#             nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
#             nn.LayerNorm([self.planes, 1, 1]),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(self.planes, self.inplanes, kernel_size=1))

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
#         # context = self.att_eca(x)

#         out = x
#         channel_add_term = self.channel_add_conv(context)
#         out = out + channel_add_term

#         return out

# class gc_module(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super(gc_module, self).__init__()
#         self.channel = GL(inplanes=in_channels)

#     @staticmethod
#     def get_module_name():
#         return "gc"

#     def forward(self, x):
#         y = self.channel(x)

#         return y