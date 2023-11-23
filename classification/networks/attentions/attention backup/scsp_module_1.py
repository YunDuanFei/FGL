import torch
from torch import nn
from .utils import DctCFea


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
        self.channel = Channelatt(in_channels, reduction=reduction)

    @staticmethod
    def get_module_name():
        return "scsp"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    print(x)
    model = MSCAttention(64)
    y = model(x)
    print(y)

