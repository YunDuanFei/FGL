import os
import re
import argparse
from datetime import datetime
from .yacs import CfgNode as CN


# -----------------------------------------------------------------------------
# global setting
# -----------------------------------------------------------------------------
cfg = CN()
cfg.run = 'run_1'
cfg.workers = 4
cfg.using_amp = True
cfg.imagenet_train_size = 1282167
cfg.imagenet_test_size = 5000
cfg.record_dir = './doc/record'
cfg.result_dir = './doc/result'
cfg.tensor_Board = './tensorboard'
cfg.time_now = datetime.now().isoformat()
cfg.dataroot = '/home/liuyunfei/data/Imagenet'
cfg.attentions = ['none', 'scsp', 'coord', 'eca', 'cbam', 'se', 'spolarized', 'fca',
                  'a2', 'gc', 'ge', 'sa', 'srm', 'triplet'
                 ]


# -----------------------------------------------------------------------------
# train
# -----------------------------------------------------------------------------
cfg.train = CN()
cfg.train.lr = 0.1
cfg.train.epoch = 100
cfg.train.optim = 'SGD'
cfg.train.momentum = 0.9
cfg.train.pretrain = False
cfg.train.log_interval = 1000
cfg.train.weight_decay = 1e-4
cfg.train.batch_size = 64
cfg.train.clip_gradient = False # for resnet 101, avoiding loss=NaN
cfg.train.max_norm = 20.0


# -----------------------------------------------------------------------------
# test
# -----------------------------------------------------------------------------
cfg.test = CN()
cfg.test.epoch = -1
cfg.test.log_interval = 100
cfg.test.batch_size = 100


def make_cfg_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
    parser.add_argument("--seed", type=int, default=0, help='seed for imagenet')
    parser.add_argument("--warm", type=int, default=5, help='warmup for imagenet')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='mobilenext_50',
                    help='model architecture')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size per process (default: 128)')
    parser.add_argument('--device', default='cuda', type=str,
                    help='model GPU train')
    parser.add_argument('--attention_type', default='scsp', type=str, choices=cfg.attentions,
                    help='attention module')
    parser.add_argument('--attention_param', default=16, type=float,
                    help='attention parameter')
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    cfg.arch = args.arch
    cfg.attention_type = args.attention_type
    cfg.record_dir = os.path.join(cfg.record_dir, cfg.arch, cfg.attention_type)
    cfg.result_dir = os.path.join(cfg.result_dir, cfg.arch, cfg.attention_type)
    if int(re.findall("\d+", args.arch)[0]) == [101,152]:
        cfg.train.clip_gradient = True


    return cfg.clone(), args