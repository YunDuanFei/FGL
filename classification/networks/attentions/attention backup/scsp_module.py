import torch
from torch import nn
import math
from torch.nn import functional as F
from .utils import SwitchNorm2d, Aconcfunc, MetaAconcfunc, DctCFea, DctSFea


class SpatialAtt(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, reduction=32):
        super(SpatialAtt, self).__init__()
        mip = max(8, in_channels // reduction)
        pad = (k_size - 1) // 2

        # pixel and channel interaction
        self.dctpool = DctSFea(in_channels, bnumfre=4)
        H, W = self.dctpool.dct_h, self.dctpool.dct_w
        self.pool_h = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w = nn.AdaptiveMaxPool2d((1, None))
        self.conv_hw = nn.Conv2d(in_channels, mip, kernel_size=(1, k_size), padding=(0, pad))
        self.register_parameter('wavgh', nn.Parameter(torch.Tensor([[[[0.5]*H]] * in_channels]).float()))
        self.register_parameter('wmaxh', nn.Parameter(torch.Tensor([[[[0.5]*H]] * in_channels]).float()))
        self.register_parameter('wavgw', nn.Parameter(torch.Tensor([[[[0.5]*W]] * in_channels]).float()))
        self.register_parameter('wmaxw', nn.Parameter(torch.Tensor([[[[0.5]*W]] * in_channels]).float()))

        # channel recover
        self.conv_h = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, out_channels, kernel_size=1, stride=1, padding=0)

        self.bn = SwitchNorm2d(mip)
        self.func = MetaAconcfunc(mip)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        x_avgh, x_avgw = self.dctpool(x)
        x_maxh, x_maxw = self.pool_h(x), self.pool_w(x)
        x_avgh, x_maxh = x_avgh.permute(0, 1, 3, 2), x_maxh.permute(0, 1, 3, 2)
        x_h, x_w = self.wavgh * x_avgh + self.wmaxh * x_maxh, self.wavgw * x_avgw + self.wmaxw * x_maxw
        y = torch.cat([x_w, x_h], dim=2)

        y = self.func(self.bn(self.conv_hw(y)))
        x_w, x_h = torch.split(y, [1, 1], dim=2)
        x_h = x_h.permute(0, 1, 3, 2)

        x_h = self.sigmoid(self.conv_h(x_h))
        x_w = self.sigmoid(self.conv_w(x_w))
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


# class Channelatt(nn.Module):
#     def __init__(self, in_channels, reduction=4):
#         super(Channelatt, self).__init__()
#         # global channel
#         self.gc = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0)
#         self.softmax = nn.Softmax(dim=-1)
#         # local channel
#         self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
#         self.dctpool = DctCFea(in_channels, bnumfre=2)
#         self.lc = nn.Conv1d(1, 1, kernel_size=3, padding=1)
#         self.lcln = nn.LayerNorm([in_channels, 1])

#         # channel attention
#         self.ca = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
#         self.ln = nn.LayerNorm(in_channels)
#         self.sigmoid = nn.Sigmoid()

#         self.register_parameter('wdct', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))
#         self.register_parameter('wmax', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))

#     def forward(self, x):
#         N, C, H, W = x.shape

#         # global
#         x_gc = self.gc(x).view(N, 1, H*W)
#         x_gc = self.softmax(x_gc).permute(0, 2, 1)
#         x_gc = torch.matmul(x.view(N, C, H*W), x_gc)
#         # self and local
#         x_sc = self.wmax * self.maxpool(x).squeeze(-1) + self.wdct * (self.dctpool(x).unsqueeze(-1))
#         x_lc = self.lc(x_sc.permute(0, 2, 1)).transpose(-1, -2)

#         att_c = self.lcln(x_gc + x_sc + x_lc).unsqueeze(-1)
#         att_c = self.ca(att_c).squeeze(-1).permute(0, 2, 1)
#         att_c = self.sigmoid(self.ln(att_c)).permute(0, 2, 1)
#         att_c = att_c.unsqueeze(-1)

#         y = x * att_c.expand_as(x)

#         return y
class Channelatt(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(Channelatt, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.mid_channels = int(in_channels * (1 / reduction))
        # self and local channel
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dctpool = DctCFea(self.in_channels, bnumfre=2)
        self.lc = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.lcln = nn.LayerNorm([self.in_channels, 1])

        # global channel transformation
        self.gct = nn.Sequential(nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1),
                                 nn.LayerNorm([self.mid_channels, 1, 1]),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(self.mid_channels, self.in_channels, kernel_size=1))

        self.register_parameter('wdct', nn.Parameter(torch.Tensor([[[0.5]] * self.in_channels]).float()))
        self.register_parameter('wmax', nn.Parameter(torch.Tensor([[[0.5]] * self.in_channels]).float()))

    def forward(self, x):
        # self and local
        x_sc = self.wmax * self.maxpool(x).squeeze(-1) + self.wdct * (self.dctpool(x).unsqueeze(-1))
        x_lc = self.lc(x_sc.permute(0, 2, 1)).transpose(-1, -2)
        # global
        att_g = self.lcln(x_sc + x_lc).unsqueeze(-1)
        att_g = self.gct(att_g)
        out = x + att_g

        return out



class scsp_module(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(scsp_module, self).__init__()
        self.channel = Channelatt(in_channels)

    @staticmethod
    def get_module_name():
        return "scsp"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    model = Channelatt(64)
    y = model(x)
    print(y.shape)


