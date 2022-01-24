from __future__ import print_function
import numpy as np
import shutil
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix
import itertools
from collections import OrderedDict
from .ActQuant import *
import math

STYLE='seaborn'
CMAP='RdBu'
TYPE='contour'
plt.style.use(STYLE)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 12


def lr_scheduler(args):
    cur_lr = args.cur_lr if hasattr(args, 'cur_lr') else args.lr
    epoch=args.epoch
    if (args.warmup_epochs is not None) and (epoch < (args.warmup_epochs)):
        cur_lr = (epoch + 1) * args.lr / args.warmup_epochs
    elif args.lr_scheduler == 'step':
        lr_decay=[80,160,300]
        if epochs in lr_decay:
            cur_lr=cur_lr/10
    elif args.lr_scheduler == 'exp':                                   # exponential
        cur_lr = args.lr * (args.lr_scale ** (epoch//args.lr_step_size))
    elif args.lr_scheduler == 'cosine':           # cosine
        if epoch == 0:
            cur_lr = args.lr
        else:
            lr_min = 0
            cur_lr = (args.lr - lr_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2.0  + lr_min
    else:
        ValueError('Unknown scheduler {}'.format(args.lr_scheduler))
    args.cur_lr=cur_lr
    return cur_lr


def lr_scheduler_gran(lr,cur_lr,nbatch,nepoch,batch,epoch,lr_scheduler,warmup_epochs=1):
    curr_batch=epoch*nbatch+batch
    tot_batch=nepoch*nbatch
    cur_lr = lr
    if (warmup_epochs is not None) and (epoch < (warmup_epochs)):
        cur_lr = (curr_batch + 1) * lr / (warmup_epochs*nbatch)
    elif lr_scheduler == 'step':
        lr_decay=[5,8]
        if epochs in lr_decay:
            cur_lr=cur_lr/10
    elif lr_scheduler == 'cosine':           # cosine
        lr_min = 0
        cur_lr = (lr - lr_min) * (1 + math.cos(math.pi * curr_batch / tot_batch)) / 2.0  + lr_min
    else:
        ValueError('Unknown scheduler {}'.format(lr_scheduler))
    return cur_lr

def accuracy(output, target, topk=1):
    """Computes the precision@k for the specified values of k"""
    maxk = max((topk,))
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    k=topk
    correct_k = correct[:k].view(-1).float().sum(0)
    return correct_k

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def toTrain(model):
    model.train()

def toEval(model):
    model.eval()

def toCuda(model):
    model.cuda()
    for p in model.parameters():
        if hasattr(p,'fp'):
            p.fp=p.fp.cuda()
            p.quant=p.quant.cuda()

## Checkpoint
def save_checkpoint(model,args,name=None):
    for p in list(model.parameters()):
        if hasattr(p,'fp'):
            p.data.copy_(p.fp)
    param_dict=model.state_dict()
    quant_dict={}
    for k,p in model.named_parameters():
        if hasattr(p,'fp'):
            quant_dict.update({k:p.quant})
    act_bits={}
    act_clip={}
    for n,m in model.named_modules():
        if isinstance(m, ActQuantizer):
            act_bits.update({n:m.num_bits})
            act_clip.update({n:m.clip_init})

    save_state={'param':param_dict,
    'quant':quant_dict,
    'epoch':args.epoch+1,
    'quantize':args.quantize,
    'lr':args.lr,
    'act_bits':act_bits,
    'act_clip':act_clip
    }

    if name==None:
        filename=os.path.join(args.log_dir,'Savedmodel{}.pt'.format(args.epoch))
    else:
        filename=os.path.join(args.log_dir,name)
    torch.save(save_state,filename)

def load_checkpoint(model,args,no_quant_update=False):
    state=torch.load(args.checkpoint)
    model.load_state_dict(state['param'])
    quant_dict=state['quant']
    if 'act_bits' in state:
        act_bits=state['act_bits']
        for n,m in model.named_modules():
            if isinstance(m, ActQuantizer):
                if n in act_bits:
                    m.num_bits=    act_bits[n]

    if 'act_clip' in state:
        act_clip=state['act_clip']
        for n,m in model.named_modules():
            if isinstance(m, ActQuantizer):
                if n in act_clip:
                    m.clip_init=act_clip[n]

    for k,p in model.named_parameters():
        if hasattr(p,'fp'):
            p.fp.copy_(p.data)
            if not no_quant_update:
                p.quant=quant_dict[k].clone()
    args.epoch=state['epoch']
    args.lr=state['lr']


def costs_mobilenet(model):
    model_size=0
    param_size=0
    param_size_list=[]
    prec_list=[]
    comp_cost=[]
    names=[]
    count=0
    prec_list_act, param_size_list_act,names_act = costs_mobilenet_act(model)
    count_k=0
    for n,p in model.named_parameters():
        if hasattr(p,'quant'):
            print("-------------------")
            conv_size=1
            for dim in p.data.shape:
                conv_size=dim*conv_size
            model_size=model_size+conv_size*p.quant.mean()
            param_size+=conv_size
            param_size_list.append(conv_size)
            prec_list.append(p.quant.mean().cpu().numpy().tolist())
            if '.0.0.weight' in n:
                names.append('FL')
                print(names_act[count_k], "|", param_size_list_act[count_k]/p.data.shape[1] ,"|", prec_list_act[count_k])
                compute_per_quant = (param_size_list_act[count_k+1]/p.data.shape[0])*p.data.shape[1]*p.data.shape[2]*p.data.shape[3]
                compute_per_layer = np.sum(p.quant.cpu().numpy()*prec_list_act[count_k]*compute_per_quant)
            elif '.0.weight' in n:
                count=count+1
                print(names_act[count_k], "|", param_size_list_act[count_k]/p.data.shape[0] ,"|", prec_list_act[count_k])
                names.append('DW{}'.format(count))
                compute_per_quant = (param_size_list_act[count_k+1]/p.data.shape[0])*p.data.shape[1]*p.data.shape[2]*p.data.shape[3]
                compute_per_layer = np.sum(p.quant.cpu().numpy()*prec_list_act[count_k]*compute_per_quant)
            elif '.3.weight' in n:
                print(names_act[count_k], "|", param_size_list_act[count_k]/p.data.shape[1] ,"|", prec_list_act[count_k])
                names.append('PW{}'.format(count))
                compute_per_quant = (param_size_list_act[count_k]/p.data.shape[1])*p.data.shape[1]*p.data.shape[2]*p.data.shape[3]
                compute_per_layer = np.sum(p.quant.cpu().numpy()*prec_list_act[count_k]*compute_per_quant)
            elif 'classifier' in n:
                print(names_act[count_k], "|", param_size_list_act[count_k]/p.data.shape[1] ,"|", prec_list_act[count_k])
                compute_per_quant = p.data.shape[1]
                compute_per_layer = np.sum(p.quant.cpu().numpy()*prec_list_act[count_k]*compute_per_quant)
                names.append('FC'.format(count))
            comp_cost.append(compute_per_layer)
            print(n, " : ", p.data.shape," : " , compute_per_layer," : ",conv_size)
            count_k = count_k + 1
            

    summary = {'Activation Size': np.sum(np.array(param_size_list_act)*np.array(prec_list_act))/(10**7),
                'Model Size' : model_size/(10**7),
                'Compute Size' : np.sum(np.array(comp_cost))/(10**10),
                'Average Bits per weight' : float(model_size)/float(param_size)}

    print('Activation Size: {}'.format(np.sum(np.array(param_size_list_act)*np.array(prec_list_act))/(10**7)))
    print('Model Size: {}'.format(model_size/(10**7)))
    print('Compute Size: {}'.format(np.sum(np.array(comp_cost))/(10**10)))
    print('Average Bits per weight: {}'.format(float(model_size)/float(param_size)))
    return prec_list,param_size_list,names,comp_cost,summary



def costs_mobilenet_act(model):
    model_size=0
    param_size=0
    param_size_list=[]
    prec_list=[]
    names=[]
    count=0
    param_size_list.append(150528)
    prec_list.append(8)
    for n,m in model.named_modules():
        if isinstance(m, ActQuantizer):
            model_size+=m.num_bits*m.sizetrack
            param_size+=m.sizetrack
            param_size_list.append(m.sizetrack)
            prec_list.append(m.num_bits)
            if '.0.2' in n:
                names.append('FL')
            elif '.2' in n:
                count=count+1
                names.append('DW{}'.format(count))
            elif '.5' in n:
                names.append('PW{}'.format(count))
    names.append('FC'.format(count))
    print('Model Size: {}'.format(model_size))
    print('Average Bits per activation: {}'.format(float(model_size)/float(param_size)))
    return prec_list,param_size_list,names

def plotsizemobilenet(model,type):
    if type=='weight':
        prec,size,names,compute,summary=costs_mobilenet(model)
    else:
        prec,size,names=costs_mobilenet_act(model)
    #fig = plt.figure()#figsize=[5,6])

    plt.clf()
    ax2 = plt.subplot(211)
    ax2.bar(1+np.arange(len(size)),prec,tick_label=names)
    plt.xlabel('Layer Index')
    plt.ylabel('Average bit-precision')
    plt.xticks(rotation=90)

    ax1 = plt.subplot(212)
    ax1.bar(1+np.arange(len(size)),size,tick_label=names)
    plt.xlabel('Layer Index')
    plt.ylabel(r'Number of parameters')
    if type=='weight':
        plt.yscale('log')
    else:
        ax1.ticklabel_format(axis='y',style='sci',scilimits=(5,5),useMathText='True')
    plt.xticks(rotation=90)
    plt.tight_layout()

    if type=='weight':
        outdict={'prec':prec,'size':size,'names':names,'compute':compute, 'summary':summary}
    else:
        outdict={'prec':prec,'size':size,'names':names}
    return outdict

def plotsize(size,prec):
    title='CIFAR-10 on ResNet-20'

    #fig = plt.figure()#figsize=[5,6])
    ax1 = plt.subplot(211)
    plt.title(title, fontname='Helvetica', fontsize=16, fontweight='bold')
    ax1.bar(1+np.arange(len(size)),size)
    plt.xlabel('Layer Index')
    plt.ylabel('Number of parameters')

    ax2 = plt.subplot(212)
    ax2.bar(1+np.arange(len(size)),prec)
    plt.xlabel('Layer Index')
    plt.ylabel('Average bit-precision')



## creates a confusion matrix
def show_confusion_matrix(predictions,targets,classes,normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure()
    cm= confusion_matrix(targets, predictions)
    if normalize:
        cm = cm.astype('fp') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return cm

## Plots the learning curve files given a folder
def plot_learning(LogFolder):
    Train_data=np.loadtxt(os.path.join(LogFolder,'LearningCurveTrain.txt'),skiprows=1)
    Test_data=np.loadtxt(os.path.join(LogFolder,'LearningCurveTest.txt'),skiprows=1)


    fig = plt.figure(figsize=[5,6])
    ax1 = plt.subplot(211)
    ax1.set_title('Learning Curves on Train Data', fontname='Ubuntu', fontsize=16,
            fontstyle='italic', fontweight='bold')
    ax1.plot(Train_data[:,0], Train_data[:,1], linewidth=1, linestyle='-', marker='o', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function')
    ax2 = plt.subplot(212)
    ax2.plot(Train_data[:,0], 1-np.squeeze((Train_data[:,2])).astype(float)/np.squeeze(Train_data[:,3]), linewidth=1, linestyle='-', marker='o',markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    ax1.grid(color='black', linestyle='-', linewidth=0.5)
    ax2.grid(color='black', linestyle='-', linewidth=0.5)
    fig.savefig(os.path.join(LogFolder,'LearningTrain.png'), bbox_inches='tight')

    fig = plt.figure(figsize=[5,6])
    ax1 = plt.subplot(211)
    ax1.set_title('Learning Curves on Test Data', fontname='Ubuntu', fontsize=16,
            fontstyle='italic', fontweight='bold')
    ax1.plot(Test_data[:,0], Test_data[:,1], linewidth=1, linestyle='-', marker='o', markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Function')
    ax2 = plt.subplot(212)
    ax2.plot(Test_data[:,0], 1-np.squeeze((Test_data[:,2])).astype(float)/np.squeeze(Test_data[:,3]), linewidth=1, linestyle='-', marker='o',markersize=2)
    plt.xlabel('Epoch')
    plt.ylabel('Error Rate')
    ax1.grid(color='black', linestyle='-', linewidth=0.5)
    ax2.grid(color='black', linestyle='-', linewidth=0.5)
    fig.savefig(os.path.join(LogFolder,'LearningTest.png'), bbox_inches='tight')
