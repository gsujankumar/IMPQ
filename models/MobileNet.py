import torch
import torch.nn as nn
from utils.QuantModules import QuantizeLinear, QuantizeConv2d
import numpy as np



class MobileNet(nn.Module):
    def __init__(self, nclass=10 ,Quantize='Dont',ActQuant=False):
        super(MobileNet, self).__init__()
        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                QuantizeConv2d(Quantize,inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride,expansion=1):
#                if True:
            if expansion==1:
                return nn.Sequential(

                    QuantizeConv2d(Quantize,inp, inp*expansion, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp*expansion),
                    nn.ReLU(inplace=True),

                    QuantizeConv2d(Quantize,inp*expansion, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )
            else:
                return nn.Sequential(

                    QuantizeConv2d(Quantize,inp, inp*expansion, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(inp*expansion),
                    nn.ReLU(inplace=True),

                    QuantizeConv2d(Quantize,inp*expansion, inp*expansion, 3, stride, 1, groups=inp*expansion, bias=False),
                    nn.BatchNorm2d(inp*expansion),
                    nn.ReLU(inplace=True),

                    QuantizeConv2d(Quantize,inp*expansion, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )

        self.features = nn.Sequential(
            conv_bn(  3,  32, 2),
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            #nn.AvgPool2d(7),
        )
        self.classifier = QuantizeLinear(Quantize, 1024, nclass)
        #self.classifier = nn.Linear(1024, nclass)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(3).mean(2)
        x = x.view(-1, 1024)
        x = self.classifier(x)
        return x
