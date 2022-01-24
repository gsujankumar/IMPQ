import torch
import torch.nn as nn
from utils.QuantModules import QuantizeLinear, QuantizeConv2d
import numpy as np

class VGGSmall(nn.Module):
    def __init__(self, Nclass=10,quantize='Dont'):
        super(VGGSmall, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                QuantizeConv2d(quantize,inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )
        def conv_mp(inp, oup, pool):
            return nn.Sequential(
                QuantizeConv2d(quantize,inp, oup, 3, 1, 1, bias=False),
                nn.MaxPool2d(pool),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        self.model = nn.Sequential(
            conv_bn(  3,  128, 1),
            conv_mp(128, 128 , 2),
            conv_bn(128, 256 , 1),
            conv_mp(256, 256 , 2),
            conv_bn(256, 512 , 1),
            conv_mp(512, 512 , 2)
        )
        self.fc = QuantizeLinear(quantize, 512*4*4, Nclass)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512*4*4)
        x = self.fc(x)
        return x
