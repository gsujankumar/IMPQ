from __future__ import print_function
import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np
import itertools

def AnalyseNet(model):
    i=1
    size=[]
    prec=[]
    for p in model.parameters():
        if hasattr(p,'quant'):
            conv_size=1
            for dim in p.data.shape:
                conv_size=dim*conv_size
            print('{} : Number of parameters : {}'.format(i,conv_size))
            print('{} : Average Bit precision : {}'.format(i,p.quant.mean() ))
            size.append(conv_size)
            prec.append(p.quant.mean())
            i=i+1
    return size,prec

def get_evalues(model,loader,batchpct=0.5,ntop=5):
    model.zero_grad()
    print(len(loader))
    nbatch=int(len(loader)*batchpct/100)
    print('Calculating E Values ({}/{}):'.format(0,nbatch))
    print('-', end = '')
    for batch_idx, (data, target) in enumerate(loader):
        #print(batch_idx)
        if next(model.parameters()).is_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output=model(data)
        max,pred=output.max(dim=1)
        diff=output.add(-max.unsqueeze(1))
        for i in range(output.shape[0]):
            sort_idx=torch.argsort(-diff[i,:])
            for count in range(1,ntop):
                k=sort_idx[count]
                model.zero_grad()
                if diff[i,k] != 0:
                    diff[i,k].backward(retain_graph=True)
                    with torch.no_grad():
                        for p in model.parameters():
                            if hasattr(p,'evalue'):
                                p.evalue=p.evalue+p.grad.data.pow(2).div(diff[i,k].pow(2))
                            else:
                                p.evalue=p.grad.data.clone().pow(2).div(diff[i,k].pow(2))

        if batch_idx==nbatch:
            model.zero_grad()
            break
        if int((batch_idx+1)/nbatch*50)>int((batch_idx)/nbatch*50):
            print('-', end = '')
    print('')

def zero_evalues(model):
    for p in model.parameters():
        p.evalue.mul_(0)



def reduce_precision(model,N_Remove=10,quantize='Lin'):
    flag=False
    for p in model.parameters():
        if hasattr(p,'quant'):
            p.metric=p.evalue.view(p.data.shape[0],-1).mean(dim=1).mul(p.data.abs().view(p.data.shape[0],-1).mean(dim=1)).div(torch.pow(2,p.quant))
            if flag:
                all_metric=torch.cat((p.metric,all_metric),0)
                all_quant=torch.cat((p.quant,all_quant),0)
            else:
                all_metric=p.metric.clone()
                all_quant=p.quant.clone()
                flag=True
    sorted_metric,_=torch.sort(all_metric[all_quant>1])
    metric_threshold=sorted_metric[N_Remove]
    for p in model.parameters():
        if hasattr(p,'quant'):
            p.quant[p.metric<=metric_threshold]-=1
            p.quant[p.quant<1]=1


def model_size(model):
    model_size=0
    param_size=0
    for p in model.parameters():
        if hasattr(p,'quant'):
            conv_size=1
            for dim in p.data.shape:
                conv_size=dim*conv_size
            model_size=model_size+conv_size*p.quant.mean()
            param_size+=conv_size
    print('Model Size: {}'.format(model_size))
    print('Average Bits per weight: {}'.format(float(model_size)/float(param_size)))
    return model_size,param_size


def LinQuant(input,quant,random=False):
    #meanscale=input.view(input.shape[0],-1).abs().mean(dim=1).mul(4).unsqueeze(1)
    meanscale,_=input.view(input.shape[0],-1).abs().mul(2).max(dim=1)
    meanscale=meanscale.unsqueeze(1)
    quant_levels=torch.pow(2,quant).unsqueeze(1)
    if len(input.size())==4:
        meanscale=meanscale.unsqueeze(2).unsqueeze(3)
        quant_levels=quant_levels.unsqueeze(2).unsqueeze(3)
    output_s=input.clone().div(meanscale).clamp(max=(0.5-0.000001)).clamp(min=(-0.5+0.000001)).mul(quant_levels)
    output=output_s.add(-0.5).round()
    if random:
        diff=output_s-output
        rand_noise=torch.rand(output.shape).type(output.type())
        diff_add=diff.clone().mul(0)
        diff_add[(rand_noise-diff)>0]=1
        ouptut=output+diff
    output=output.add(0.5).div(quant_levels).mul(meanscale)
    return output


class QuantizeLinear(nn.Linear):

    def __init__(self,quantize,*kargs, **kwargs):
        super(QuantizeLinear, self).__init__(*kargs, **kwargs)
        self.quantize=quantize
        self.weight.fp=self.weight.data.clone()
        self.weight.quant=torch.ones([self.weight.shape[0]]).type(self.weight.data.type())
    def forward(self, input):
        if self.quantize=='Lin':
            self.weight.data=LinQuant(self.weight.fp,self.weight.quant)
        out = nn.functional.linear(input, self.weight)
        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)
        return out
    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        ) + ', quantize={}, quant={}'.format(self.quantize,self.weight.quant.mean())

class QuantizeConv2d(nn.Conv2d):

    def __init__(self,quantize, *kargs, **kwargs):
        super(QuantizeConv2d, self).__init__(*kargs, **kwargs)
        self.quantize=quantize
        self.weight.fp=self.weight.data.clone()
        self.weight.quant=torch.ones([self.weight.shape[0]]).type(self.weight.data.type())
    def forward(self, input):
        if self.quantize=='Lin':
            self.weight.data=LinQuant(self.weight.fp,self.weight.quant)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
                                   self.padding, self.dilation, self.groups)
        if not self.bias is None:
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out
    def extra_repr(self):

        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__) + ', quantize={}, quant={}'.format(self.quantize,self.weight.quant.mean())
