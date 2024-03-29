###################################################################################################################
# code reference https://github.com/DeLightCMU/PSA/blob/main/semantic-segmentation/network/PSA.py
# Paper: `Polarized Self-Attention: Towards High-quality Pixel-wise Regression` - https://arxiv.org/abs/2107.00782
###################################################################################################################

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F


# class PSA_p(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=1, stride=1):
#         super(PSA_p, self).__init__()

#         self.inplanes = inplanes
#         self.inter_planes = planes // 2
#         self.planes = planes
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = (kernel_size-1)//2

#         self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
#         self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
#         self.softmax_right = nn.Softmax(dim=2)
#         self.sigmoid = nn.Sigmoid()

#         self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #g
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
#         self.softmax_left = nn.Softmax(dim=2)

#     def spatial_pool(self, x):
#         input_x = self.conv_v_right(x)

#         batch, channel, height, width = input_x.size()

#         # [N, IC, H*W]
#         input_x = input_x.view(batch, channel, height * width)

#         # [N, 1, H, W]
#         context_mask = self.conv_q_right(x)

#         # [N, 1, H*W]
#         context_mask = context_mask.view(batch, 1, height * width)

#         # [N, 1, H*W]
#         context_mask = self.softmax_right(context_mask)

#         # [N, IC, 1]
#         # context = torch.einsum('ndw,new->nde', input_x, context_mask)
#         context = torch.matmul(input_x, context_mask.transpose(1,2))
#         # [N, IC, 1, 1]
#         context = context.unsqueeze(-1)

#         # [N, OC, 1, 1]
#         context = self.conv_up(context)

#         # [N, OC, 1, 1]
#         mask_ch = self.sigmoid(context)

#         out = x * mask_ch

#         return out

#     def channel_pool(self, x):
#         # [N, IC, H, W]
#         g_x = self.conv_q_left(x)

#         batch, channel, height, width = g_x.size()

#         # [N, IC, 1, 1]
#         avg_x = self.avg_pool(g_x)

#         batch, channel, avg_x_h, avg_x_w = avg_x.size()

#         # [N, 1, IC]
#         avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

#         # [N, IC, H*W]
#         theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

#         # [N, 1, H*W]
#         # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
#         context = torch.matmul(avg_x, theta_x)
#         # [N, 1, H*W]
#         context = self.softmax_left(context)

#         # [N, 1, H, W]
#         context = context.view(batch, 1, height, width)

#         # [N, 1, H, W]
#         mask_sp = self.sigmoid(context)

#         out = x * mask_sp

#         return out

#     def forward(self, x):
#         # [N, C, H, W]
#         context_channel = self.spatial_pool(x)
#         # [N, C, H, W]
#         context_spatial = self.channel_pool(x)
#         # [N, C, H, W]
#         out = context_spatial + context_channel
#         return out

class PSA_s(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_s, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        ratio = 4

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                      bias=False)
        # self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.inter_planes, self.inter_planes // ratio, kernel_size=1),
            nn.LayerNorm([self.inter_planes // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_planes // ratio, self.planes, kernel_size=1)
        )
        self.softmax_right = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0,
                                     bias=False)  # theta
        self.softmax_left = nn.Softmax(dim=2)

    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)

        batch, channel, height, width = input_x.size()

        # [N, IC, H*W]
        input_x = input_x.view(batch, channel, height * width)

        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)

        # [N, 1, H*W]
        context_mask = context_mask.view(batch, 1, height * width)

        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)

        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))

        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)

        # [N, OC, 1, 1]
        context = self.conv_up(context)

        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)

        out = x * mask_ch

        return out

    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)

        batch, channel, height, width = g_x.size()

        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)

        batch, channel, avg_x_h, avg_x_w = avg_x.size()

        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)

        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)

        # [N, IC, H*W]
        theta_x = self.softmax_left(theta_x)

        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)

        # [N, 1, H, W]
        context = context.view(batch, 1, height, width)

        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)

        out = x * mask_sp

        return out

    def forward(self, x):
        # [N, C, H, W]
        out = self.spatial_pool(x)

        # [N, C, H, W]
        out = self.channel_pool(out)

        # [N, C, H, W]
        # out = context_spatial + context_channel

        return out

class spolarized_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(spolarized_module, self).__init__()
        self.channelspatial = PSA_s(in_channels, in_channels)

    @staticmethod
    def get_module_name():
        return "spolarized"

    def forward(self, x):
        y = self.channelspatial(x)

        return y