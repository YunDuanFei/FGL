###########################################################################################
# code reference https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
# Paper: `Squeeze-and-Excitation Networks` - https://arxiv.org/abs/1709.01507
###########################################################################################

from torch import nn

class se_module(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(se_module, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), in_channels, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def get_module_name():
        return "se"

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y