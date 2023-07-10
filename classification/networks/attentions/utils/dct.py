import torch
from torch import nn
import math


BACKBONES = {
    'resnet18': dict([(64, 56), (128, 28), (256, 14), (512, 7)]),
    'resnet34': dict([(64, 56), (128, 28), (256, 14), (512, 7)]),
    'resnet50': dict([(256, 56), (512, 28), (1024, 14), (2048, 7)]),
    'resnet101': dict([(256, 56), (512, 28), (1024, 14), (2048, 7)]),
    'resnet152': dict([(256, 56), (512, 28), (1024, 14), (2048, 7)]),
    'mobilenet_100': dict([(24, [(96, 56), (144, 56)]),
        (32, [(144, 28), (192, 28), (192, 28)]),
        (64, [(192, 14), (384, 14), (384, 14), (384, 14)]),
        (96, [(384, 14), (576, 14), (576, 14)]),
        (160, [(576, 7), (960, 7), (960, 7)]),
        (320, [(960, 7)])
        ]),
    'mobilenet_75': dict([(24, [(96, 56), (144, 56)]),
        (32, [(144, 28), (144, 28), (144, 28)]),
        (64, [(144, 14), (288, 14), (288, 14), (288, 14)]),
        (96, [(288, 14), (432, 14), (432, 14)]),
        (160, [(432, 7), (720, 7), (720, 7)]),
        (320, [(720, 7)])
        ]),
    'mobilenet_50': dict([(24, [(48, 56), (96, 56)]),
        (32, [(96, 28), (96, 28), (96, 28)]),
        (64, [(96, 14), (192, 14), (192, 14), (192, 14)]),
        (96, [(192, 14), (288, 14), (288, 14)]),
        (160, [(288, 7), (480, 7), (480, 7)]),
        (320, [(480, 7)])
        ]),
    "mobilenext_100": dict([(96, 56), (192, 28), (288, 14), (384, 14), (576, 7), (960, 7)]),
    "mobilenext_75": dict([(72, 56), (144, 28), (216, 14), (288, 14), (432, 7), (720, 7)]),
    "mobilenext_50": dict([(48, 56), (96, 28), (144, 14), (192, 14), (288, 7), (480, 7)]),
}


class DctCFea(nn.Module):
    def __init__(self, in_channels, bnumfre=4, eps=1e-7, backbone='resnet18', stage=None):
        super(DctCFea, self).__init__()
        if backbone.split('_')[0] == 'mobilenet':
            stage = stage.split('_')
            last_channel = int(stage[0])
            mid_stage = int(stage[1])
            c2wh = BACKBONES[backbone][last_channel][mid_stage]
            self.dct_h, self.dct_w = c2wh[1], c2wh[1]
            in_channels = c2wh[0]
        else:
            c2wh = BACKBONES[backbone]
            self.dct_h, self.dct_w = c2wh[in_channels], c2wh[in_channels]
            in_channels = in_channels

        self.numfre = int(in_channels // bnumfre)

        self.register_parameter('basicfrex', nn.Parameter(torch.rand(self.numfre, 1)))
        self.register_parameter('basicfrey', nn.Parameter(torch.rand(self.numfre, 1)))

        def intfre(bfre):
            freamp = 7 / (torch.max(bfre) + eps)
            bfre = bfre * freamp
            bfre = torch.floor(torch.clamp(bfre, 0, 6))
            return bfre.squeeze(-1).detach().numpy().astype(dtype=int).tolist()

        mapper_x, mapper_y = intfre(self.basicfrex), intfre(self.basicfrey)
        mapper_x = [temp_x * (self.dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (self.dct_w // 7) for temp_y in mapper_y]

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(self.dct_h, self.dct_w, mapper_x, mapper_y, in_channels))
        # learnable DCT init
        # self.register_parameter('weight', nn.Parameter(self.get_dct_filter(height, width, mapper_x, mapper_y, channel)))

    def forward(self, x):
        n, c, h, w = x.shape
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        if h != self.dct_h or w != self.dct_w:
            x = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        x = x * self.weight
        x = torch.sum(x, dim=[2, 3])

        return x

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)
        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter
