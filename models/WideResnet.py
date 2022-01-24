import torch
import torch.nn as nn
from torch.nn import init
from utils.QuantModules import QuantizeLinear, QuantizeConv2d
import numpy as np
import math

def conv3x3(in_planes, out_planes, stride=1,quantize='Dont'):
    """3x3 convolution with padding"""
    return QuantizeConv2d(quantize,in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes,quantize='Dont'):
    """3x3 convolution with padding"""
    return QuantizeConv2d(quantize,in_planes, out_planes, kernel_size=1, stride=1,
                     padding=0, bias=False)



class DownsampleA(nn.Module):

    def __init__(self, nIn, nOut, stride,quantize):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize='Dont'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride,quantize)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes,1,quantize)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, quantize='Dont'):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv1x1(inplanes, planes,quantize)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv3x3(planes, planes,stride,quantize)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = Conv1x1(planes, planes * self.expansion,quantize)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu3(out)

        return out

class CifarWideResNet(nn.Module):
    """
    ResNet optimized for the Cifar dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """
    def __init__(self, block, depth, num_classes,quantize='Dont'):
        super(CifarResNet, self).__init__()
        #Model type specifies number of layers for CIFAR-10 and CIFAR-100 model
        assert (depth - 2) % 6 == 0, 'depth should be one of 20, 32, 44, 56, 110'
        layer_blocks = (depth - 2) // 6
        print ('CifarResNet : Depth : {} , Layers for each block : {}'.format(depth, layer_blocks))

        self.num_classes = num_classes
        self.quantize=quantize
        self.conv_1_3x3 = conv3x3(3, 16, stride=1,quantize=self.quantize)
        self.bn_1 = nn.BatchNorm2d(16)
        self.relu_1 = nn.ReLU(inplace=True)
        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = QuantizeLinear(self.quantize,64*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, QuantizeConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, QuantizeLinear):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride,self.quantize)
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,self.quantize))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,quantize=self.quantize))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1_3x3(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def wideresnet20(num_classes=10,quantize='Dont'):
    """Constructs a ResNet-20 model for CIFAR-10 (by default)
    Args:
    num_classes (uint): number of classes
    """
    model = CifarWideResNet(BasicBlock, 20, num_classes,quantize)
    return model
