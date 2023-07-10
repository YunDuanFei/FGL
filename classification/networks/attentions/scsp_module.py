import torch
from torch import nn
from .utils import DctCFea


class Channelatt(nn.Module):
    def __init__(self, in_channels, reduction=4, backbone='resnet18', stage=None):
        super(Channelatt, self).__init__()
        self.in_channels = in_channels
        # global channel
        self.gc = nn.Conv2d(self.in_channels, 1, kernel_size=1, stride=1, padding=0)
        self.softmax = nn.Softmax(dim=-1)
        # local channel
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.dctpool = DctCFea(self.in_channels, bnumfre=4, backbone=backbone, stage=stage)
        self.lc = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.lcln = nn.LayerNorm([self.in_channels, 1])
        # transformation weights
        self.tw = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, groups=1)
        self.twln = nn.LayerNorm([self.in_channels, 1, 1])
        self.sigmoid = nn.Sigmoid()

        self.register_parameter('wdct', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))
        self.register_parameter('wmax', nn.Parameter(torch.Tensor([[[0.5]] * in_channels]).float()))

    def forward(self, x):
        N, C, H, W = x.shape

        # global
        x_g = self.gc(x).view(N, 1, H*W)
        x_g = self.softmax(x_g).permute(0, 2, 1)
        x_g = torch.matmul(x.view(N, C, H*W), x_g)
        # self and local
        x_s = self.wmax * self.maxpool(x).squeeze(-1) + self.wdct * (self.dctpool(x).unsqueeze(-1))
        x_l = self.lc(x_s.permute(0, 2, 1)).transpose(-1, -2)
        # attention weights
        att_c = self.lcln(x_g + x_s + x_l).unsqueeze(-1)
        att_c = self.sigmoid(self.twln(self.tw(att_c)))

        y = x * att_c.expand_as(x)

        return y


class scsp_module(nn.Module):
    def __init__(self, in_channels, reduction=4, backbone='resnet18', stage=None):
        super(scsp_module, self).__init__()
        self.channel = Channelatt(in_channels, reduction=reduction, backbone=backbone, stage=stage)

    @staticmethod
    def get_module_name():
        return "scsp"

    def forward(self, x):
        y = self.channel(x)

        return y

if __name__ == '__main__':
    x = torch.rand(2, 64, 56, 56)
    print(x)
    model = scsp_module(64)
    y = model(x)
    print(y)

