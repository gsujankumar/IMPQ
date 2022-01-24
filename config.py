import argparse
from utils import *

def reduce_param_act(args):
    outargs={}
    if args.arch=='MobileNet':
        outargs['lr_begin']=0.0002
        outargs['gransize']=[20,5,4]
        outargs['gransteps']=[0,2,6]
        outargs['lr_steps']=[3,6]
        outargs['epochs']=10
        outargs['eval_pct']=0.5
        outargs['eval_top']=5
        outargs['nbits']=8
    elif args.arch=='ResNet20':
        outargs['lr_begin']=0.0002
        outargs['gransize']=[12,8,2]
        outargs['gransteps']=[0,1,4]
        outargs['lr_steps']=[10,20]
        outargs['epochs']=30
        outargs['eval_pct']=5
        outargs['eval_top']=10
        outargs['nbits']=6
    else:
        outargs['lr_begin']=0.0002
        outargs['gransize']=[10,4]
        outargs['gransteps']=[0,4]
        outargs['lr_steps']=[3,6]
        outargs['epochs']=10
        outargs['eval_pct']=0.5
        outargs['eval_top']=5
        outargs['nbits']=8
    args.reduce_arg=outargs
    return outargs

def reduce_param(args):
    outargs={}
    if args.arch=='MobileNet':
        outargs['lr_begin']=0.0002
        outargs['gransize']=[4000,2000]
        outargs['gransteps']=[0,4]
        outargs['lr_steps']=[3,6]
        outargs['epochs']=10
        outargs['eval_pct']=0.5
        outargs['eval_top']=5

    elif args.arch=='ResNet20':
        outargs['lr_begin']=0.0002
        outargs['gransize']=[400,200,150]
        outargs['gransteps']=[0,3,4]
        outargs['lr_steps']=[10,20]
        outargs['epochs']=30
        outargs['eval_pct']=5
        outargs['eval_top']=10

    else:
        outargs['lr_begin']=0.0002
        outargs['gransize']=[4000,2000]
        outargs['gransteps']=[0,4]
        outargs['lr_steps']=[3,6]
        outargs['epochs']=10
        outargs['eval_pct']=0.5
        outargs['eval_top']=5
    args.reduce_arg=outargs
    return outargs

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Quantization Framework')

    parser.add_argument(
    '--batch-size',
    type=int,
    default=100,
    help='Input batch size for training (default: 256)')

    parser.add_argument('--checkpoint',
    type=str,
    default='NONE',
    help= 'Test a model')

    parser.add_argument('--warmup-epochs',
    default=5,
    type=int,
    help='# of epochs for learning rate warmup')

    parser.add_argument('--lr-scheduler',
    type=str,
    default='cosine',
    help= 'LR scheduler options (step, cosine, exp)')

    parser.add_argument('--lr-step-size',
    default=10,
    type=int,
    help='number of epochs to wait before reducing lr')

    parser.add_argument('--lr-scale',
    default=0.9,
    type=float,
    help='scale that reduces the learning rate')

    parser.add_argument('--bn-clip-k',
    default=6,
    type=float,
    help='Multiples of BN to clip')

    parser.add_argument('--clip-init-act',
    default=4,
    type=float,
    help='Clip Value')

    parser.add_argument('--use-bn-clip',
    default=True,
    type=bool,
    help='Activation Clipping Choice')

    parser.add_argument('--load-convert',
    type=str,
    default='NONE',
    help= 'Test a model')

    parser.add_argument(
    '--data-path',
    type=str,
    default='None',
    help='Path to dataset')

    parser.add_argument('--quantize',
    type=str,
    default='Dont',
    help= 'Weight binarization')

    parser.add_argument('--actquant-model',
    action='store_true',
    default=False,
    help= 'Quantize Activations')


    parser.add_argument('--step-freq',
    type=int,
    default=1,
    help='How often to Bpass')

    parser.add_argument('--iter-restart',
    type=int,
    default=0,
    help='Iteration number to restart')

    parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='Training status print frequency %')

    parser.add_argument(
    '--log-root',
    type=str,
    default='./LOG/',
    help='Log directory')

    parser.add_argument(
    '--log-dir',
    type=str,
    default='Junk',
    help='Log directory')

    parser.add_argument(
    '--optim',
    type=str,
    default='SGD',
    help='Optimizer')

    parser.add_argument(
    '--read-dir',
    type=str,
    default='Junk',
    help='Log directory read')

    parser.add_argument('--seed',
    type=int,
    default=10,
    help='Random seed (default: 1)')

    parser.add_argument('--lr',
    type=float,
    default=0.02,
    help='Learning rate (default: 0.001)')

    parser.add_argument('--no-cuda',
    action='store_true',
    default=False,
    help='Disables CUDA training')

    parser.add_argument('--epochs',
    type=int,
    default=400,
    help='Number of epochs')

    parser.add_argument('--arch',
    type=str,
    default='MobileNet',
    help='Model Type')

    parser.add_argument('--nbits',
    type=int,
    default=8,
    help='Number of bits')

    parser.add_argument('--mode',
    type=str,
    default='train',
    help='Choose mode train/augment/reduce')


    parser.add_argument('--dataset',
    type=str,
    default='ImageNet',
    help='Dataset')

    args = parser.parse_args()
    args.epoch=0;
    args.log_dir=os.path.join(args.log_root,args.log_dir)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.no_quant_update=False

    return args
