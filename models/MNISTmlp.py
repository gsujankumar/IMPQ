import torch
import torch.nn as nn
from utils.QuantModules import QuantizeLinear, QuantizeConv2d
import numpy as np

class MNISTmlp(nn.Module):
    def __init__(self, Nclass=10,quantize='Dont'):
        super(MNISTmlp, self).__init__()
        self.fc1 = QuantizeLinear(quantize, 784, 300)
        self.relu1=nn.ReLU(inplace=True)
        self.fc2 = QuantizeLinear(quantize, 300, 100)
        self.relu2=nn.ReLU(inplace=True)
        self.fc3 = QuantizeLinear(quantize, 100, 10)
        self.fc1_hist=[]
        self.fc2_hist=[]
    def forward(self, x):
        x = x.view(-1,28*28)
        x = self.fc1(x)
        self.fc1_hist.append(x)
        x = self.relu1(x)
        x = self.fc2(x)
        self.fc2_hist.append(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
