import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
from torch.autograd import Variable

def quantize_act_inplace(input_data, n_bits, clip):
    device = input_data.device
    input_data.clamp_(min=0)
    #b = torch.pow(torch.tensor(2.0),-n_bits).to(device)
    b = 2**(-n_bits)
    input_data/=(b*clip)
    input_data.round_()
    input_data *= b
    input_data.clamp_(max=1-b)
    input_data *=clip

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def quantize_act_inplace_signed(input_data, n_bits, clip):
    device = input_data.device
    #assume that data is already clipped
    b = 2**(1-n_bits)
    input_data/=(b*clip)
    input_data.round_()
    input_data *= b
    input_data.clamp_(max=1-b)
    input_data *=clip

class ActQuantizer(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ActQuantizer, self).__init__()
        self.clip_init = kwargs['clip_init'] #initial clip value
        self.num_bits = kwargs['num_bits'] #number of bits for the uniform quantizer on [0,clip]
        self.signed = kwargs['signed'] #choose if activations are signed or unsigned
        min_val = -self.clip_init if self.signed == True else 0
        self.relux = nn.Hardtanh(min_val=min_val, max_val=self.clip_init, inplace=True)
        self.hook= self.register_backward_hook(self.backward_hook)
        #if self.learn_clip == True:
        #    self.register_backward_hook(self.backward_hook)

    def reinit(self, clip_init, num_bits):
        if clip_init!=None:
            self.clip_init = clip_init #initial clip value
        if num_bits!=None:
            self.num_bits = num_bits #number of bits for the uniform quantizer on [0,clip]
        min_val = -self.clip_init if self.signed == True else 0
        self.relux = nn.Hardtanh(min_val=min_val, max_val=self.clip_init, inplace=True)


    def backward_hook(self, module, grad_input, grad_output):
        grad_output, = grad_output
        module.meangrad = grad_output.view(grad_output.shape[0],-1).pow(2).sum(1)

    def extra_repr(self):
        return 'clip_init={clip_init}, num_bits={num_bits}, signed={signed}'.format(**self.__dict__)

    def forward(self, input):

        self.relux.max_val = self.clip_init
        self.sizetrack=input.view(input.shape[0],-1).shape[1]
        x = self.relux(input)
        if self.signed == True:
            quantize_act_inplace_signed(x.data, self.num_bits, self.clip_init)
        else:
            quantize_act_inplace(x.data, self.num_bits, self.clip_init)

        return x

def model_size_act(model):
    model_size=0
    param_size=0
    for n,m in model.named_modules():
        if isinstance(m, ActQuantizer):
            model_size+=m.num_bits*m.sizetrack
            param_size+=m.sizetrack
    print('Model Size: {}'.format(model_size))
    print('Average Bits per weight: {}'.format(float(model_size)/float(param_size)))
    return model_size,param_size

def quantize_model(model, args):
    layer_count = 0
    relu_clip_val = args.clip_init_act
    num_bits=16
    for n,m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and args.use_bn_clip == True:
            k = args.bn_clip_k
            relu_clip_val = torch.max(m.bias.data+k*m.weight.data.abs()).item()
        if isinstance(m, nn.ReLU):
            quant_args = {'clip_init': relu_clip_val, 'learn_clip': False, 'num_bits': num_bits, 'in_place': True, 'signed': False}
            act_quant = ActQuantizer(**quant_args)
            rsetattr(model,n, act_quant)
            relu_clip_val = args.clip_init_act
#            print('Layer '+ n+ ' Replaced with ActQuantizer')


def sync_clip(model, args,nbits=None):
    layer_count = 0
    relu_clip_val = args.clip_init_act
    for n,m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) and args.use_bn_clip == True:
            k = args.bn_clip_k
            relu_clip_val = torch.max(m.bias.data+k*m.weight.data.abs()).item()
        if isinstance(m, ActQuantizer):
            m.reinit(relu_clip_val,nbits)
            relu_clip_val = args.clip_init_act

def temp_func(model,clip_init_act=4,num_bits=16):

    layer_count = 0
    relu_clip_val = -1
    for n,m in model.named_modules():
        if isinstance(m, ActQuantizer):
            m.reinit(clip_init_act,num_bits)

def zero_evalues_act(model):
    for n,m in model.named_modules():
        if isinstance(m, ActQuantizer):
            m.evalue=m.evalue*0


def reduce_precision_act(model,N_Remove=10,quantize='Lin'):
    flag=False
    all_metric=[]
    all_quant=[]
    for n,m in model.named_modules():
        if isinstance(m, ActQuantizer):
            m.metric=m.evalue.div(2**m.num_bits).mul(m.clip_init)
            all_metric.append(m.metric.cpu().numpy().tolist())
            all_quant.append(m.num_bits)

    all_metric=torch.Tensor(all_metric).cuda()
    all_quant=torch.Tensor(all_quant).cuda()
    sorted_metric,_=torch.sort(all_metric[all_quant>1])
    metric_threshold=sorted_metric[N_Remove]

    for n,m in model.named_modules():
        if isinstance(m, ActQuantizer):
            if m.metric<=metric_threshold:
                m.num_bits-=1
            if m.num_bits<1:
                m.num_bits=1
            #print(p.quant)

def get_evalues_act(model,loader,batchpct=0.5,ntop=5):
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
        sort_idx=torch.argsort(-diff,dim=1)
        for i in range(output.shape[0]):
            for count in range(1,ntop):
                model.zero_grad()
                k=sort_idx[i,count]
                if diff[i,k] != 0:
                    diff[i,k].backward(retain_graph=True)
                with torch.no_grad():
                    for n,m in model.named_modules():
                        if isinstance(m, ActQuantizer):
                            if hasattr(m,'evalue'):
                                m.evalue+=m.meangrad[i].div(diff[i,k].pow(2))
                            else:
                                m.evalue=m.meangrad[i].div(diff[i,k].pow(2))


        if batch_idx==nbatch:
            model.zero_grad()
            break
        if int((batch_idx+1)/nbatch*50)>int((batch_idx)/nbatch*50):
            print('-', end = '')
    print('')
