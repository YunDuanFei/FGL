import functools
from ..attentions import get_attention_module
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, resnet50d
from .mobilenet import mobilenet_100, mobilenet_75, mobilenet_50
from .mobilenext import mobilenext_100, mobilenext_75, mobilenext_50

model_dict = {
    "resnet18": resnet18, 
    "resnet34": resnet34, 
    "resnet50": resnet50, 
    "resnet101": resnet101, 
    "resnet152": resnet152,
    "resnet50d": resnet50d,
    "resnext50_32x4d": resnext50_32x4d,
    "mobilenet_100": mobilenet_100,
    "mobilenet_75": mobilenet_75,
    "mobilenet_50": mobilenet_50,
    "mobilenext_100": mobilenext_100,
    "mobilenext_75": mobilenext_75,
    "mobilenext_50": mobilenext_50,
}


def create_net(args, attentionall):
    net = None

    attention_module = get_attention_module(args.attention_type)

    # srm does not have any input parameters
    if args.attention_type in attentionall and args.attention_type != 'none':
        if args.attention_type in ['scsp', 'fca']:
            attention_module = functools.partial(attention_module, reduction=args.attention_param, backbone=args.arch.lower())
        else:
            attention_module = functools.partial(attention_module, reduction=args.attention_param)
        

    kwargs = {}
    kwargs["num_classes"] = 1000
    kwargs["attention_module"] = attention_module

    net = model_dict[args.arch.lower()](**kwargs)

    return net