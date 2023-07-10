###########################################################################################
# code reference https://github.com/cfzd/FcaNet/blob/master/model/layer.py
# Paper: `FcaNet: Frequency Channel Attention Networks` - https://arxiv.org/abs/2012.11879
###########################################################################################

import math
import torch
import torch.nn as nn

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


def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y

class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, in_channels, reduction = 16, freq_sel_method = 'top16', backbone='resnet18', stage=None):
        super(MultiSpectralAttentionLayer, self).__init__()
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
        
        self.reduction = reduction
        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (self.dct_h // 7) for temp_x in mapper_x] 
        mapper_y = [temp_y * (self.dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(self.dct_h, self.dct_w, mapper_x, mapper_y, in_channels)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered. 
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """
    def __init__(self, height, width, mapper_x, mapper_y, in_channels):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert in_channels % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, in_channels))

        # fixed random init
        # self.register_buffer('weight', torch.rand(in_channels, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, in_channelsvvvv))

        # learnable random init
        # self.register_parameter('weight', torch.rand(in_channels, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2,3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS) 
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, in_channels):
        dct_filter = torch.zeros(in_channels, tile_size_x, tile_size_y)

        c_part = in_channels // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i+1)*c_part, t_x, t_y] = self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter

class fca_module(nn.Module):
    def __init__(self, in_channels, reduction=4, backbone='resnet18', stage=None):
        super(fca_module, self).__init__()
        self.channel = MultiSpectralAttentionLayer(in_channels, reduction=reduction, backbone=backbone, stage=stage)

    @staticmethod
    def get_module_name():
        return "fca"

    def forward(self, x):
        y = self.channel(x)

        return y