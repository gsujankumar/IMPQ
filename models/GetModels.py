import torch
import torch.nn as nn
import utils as utils
import numpy as np

from .MobileNet import *
from .VGGSmall import *
from .ResNet import *
from .MobileNetV2 import *

def getmodels(args):
    if args.arch=='VGG':
        model = VGGSmall(args.nclass,args.quantize)
    elif args.arch=='MobileNet':
        model = MobileNet(args.nclass,args.quantize)
    elif args.arch=='MobileNetV2':
        model = mobilenetv2(num_classes=args.nclass,quantize=args.quantize)
    else:
        model = resnet20(args.nclass,args.quantize)


    #utils.quantize_model(model)
    #print(model)

    if args.cuda:
        torch.cuda.set_device(0)
        utils.toCuda(model)

    if args.actquant_model:
        utils.quantize_model(model, args)

    if args.checkpoint !='NONE':
        utils.load_checkpoint(model,args,no_quant_update=args.no_quant_update)
        if args.no_quant_update:
            for p in model.parameters():
                if hasattr(p,'quant'):
                    p.quant=p.quant.mul(0).add(args.nbits)
    else:
        for p in model.parameters():
            if hasattr(p,'quant'):
                p.quant=p.quant.mul(0).add(args.nbits)

    return model
