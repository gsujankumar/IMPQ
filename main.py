from __future__ import print_function
import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import os
import sys
import numpy as np
import models
from torch.autograd import Variable
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import config
import scipy.io as io
import pickle

args = config.get_args()
if args.mode!='evaluate':
    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)
    sys.stdout = utils.Logger(os.path.join(args.log_dir,'print.log'))
print(args)

####################### datasets #################################
train_loader,valid_loader,test_loader=utils.get_dataset(args)
####################### models #################################
model=models.getmodels(args)
print(model)
####################### optimizers #################################
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,  weight_decay=0.0005,nesterov=False)


####################### train_epoch #################################
def train(epoch,Log=False):
    utils.toTrain(model)
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.mode=='reduce' or args.mode=='actquant':
            optimizer.param_groups[0]['lr']=utils.lr_scheduler_gran(args.lr,
            optimizer.param_groups[0]['lr'],len(train_loader),
            args.reduce_arg['epochs'],batch_idx,epoch,'cosine',
            max(int(args.reduce_arg['epochs']/10),0))
        if args.mode=='actquant' and batch_idx%100==0:
            utils.sync_clip(model, args)

        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        if batch_idx % args.step_freq==0:
            if args.quantize is not 'Dont':
                for p in list(model.parameters()):
                    if hasattr(p,'fp'):
                        p.data.copy_(p.fp)
                optimizer.step()
                for p in list(model.parameters()):
                    if hasattr(p,'fp'):
                        p.fp.copy_(p.data)
            else:
                optimizer.step()

        #print loging
        if Log and int(100*batch_idx*len(data) / len(train_loader.dataset)) % args.log_interval == 0 :
            if  int(100*(batch_idx-1)*len(data) / len(train_loader.dataset)) % args.log_interval!= 0 :
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx*len(data) / len(train_loader.dataset), loss.data.item()))

####################### test_epoch #################################
def test(epoch,loader,name='Test'):
    utils.toEval(model)
    test_loss = 0
    correct = 0
    correct1 = 0
    correct5 = 0
    flag=-1
    with torch.no_grad():
        for data, target in loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            correct5 += utils.accuracy(output,target,5)
            if flag == 1:
                pred_all=torch.cat((pred_all,pred),0)
                target_all=torch.cat((target_all,target),0)
            else:
                pred_all=pred
                target_all=target
                flag=1


    test_loss = test_loss/len(loader.dataset)

    print('{} data => Epoch: {} Average loss: {:.6f}, Accuracy: {}/{} ({:.2f}%) Top5 : {:.2f}%'.format(name,
        epoch,test_loss, correct, len(loader.dataset),
        100. * float(correct) / len(loader.dataset),
        100. * float(correct5) / len(loader.dataset)))

    ## Loging
    if args.mode == 'train':
        if name=='Test':
            log_file_test.write('\n{}\t{:.8f}\t{}\t{}\t{}'.format(epoch,test_loss,correct,correct5,len(loader.dataset)))
        elif name=='Train':
            log_file_train.write('\n{}\t{:.8f}\t{}\t{}\t{}'.format(epoch,test_loss,correct,correct5,len(loader.dataset)))
    return pred_all.cpu().numpy(), target_all.cpu().numpy(), correct

####################### Code Modes #################################
if args.mode == 'actquant':
    _,_,correct=test(args.epoch,test_loader,'Test')
    reduce_arg=config.reduce_param_act(args)
    if not args.actquant_model:
        utils.quantize_model(model, args)
        utils.sync_clip(model, args,reduce_arg['nbits'])
        _,_,correct=test(args.epoch,test_loader,'Test')
        correct_old=correct
    for i in range(args.iter_restart,200):
        print('Iternation: {}'.format(i))
        optimizer.param_groups[0]['lr']=reduce_arg['lr_begin']
        args.lr=reduce_arg['lr_begin']
        print('Learning Rate: {}'.format(args.lr))
        for k in range(len(reduce_arg['gransteps'])):
            if reduce_arg['gransteps'][k]==i:
                gran=reduce_arg['gransize'][k]
        t = time.time()
        utils.model_size_act(model)
        utils.save_checkpoint(model,args,'Reduced_Model{}.pt'.format(i))
        utils.get_evalues_act(model,valid_loader,reduce_arg['eval_pct'],reduce_arg['eval_top'])
        utils.reduce_precision_act(model,gran,args.quantize)
        utils.zero_evalues_act(model)
        _,_,correct=test(0,test_loader,'Test')
        best_correct=0
        for j in range(reduce_arg['epochs']):
            elapsed = time.time() - t
            print('Time Taken: {}'.format(elapsed))
            t = time.time()
            train(j, Log=True)
            utils.sync_clip(model, args)
            _,_,correct=test(j,test_loader,'Test')
            utils.save_checkpoint(model,args,'Checkpoint{}.pt'.format(i))
            if correct>best_correct:
                utils.save_checkpoint(model,args,'Best{}.pt'.format(i))
                best_correct=correct
            if j in reduce_arg['lr_steps']:
                #optimizer.param_groups[0]['lr']=args.lr/10
                #args.lr=args.lr/10
                args.lr=args.lr
            print('Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))
        correct_old=correct


elif args.mode == 'reduce':
    _,_,correct=test(args.epoch,test_loader,'Test')
    reduce_arg=config.reduce_param(args)
    for i in range(200):
        optimizer.param_groups[0]['lr']=reduce_arg['lr_begin']
        args.lr=reduce_arg['lr_begin']
        print('Learning Rate: {}'.format(args.lr))
        for k in range(len(reduce_arg['gransteps'])):
            if reduce_arg['gransteps'][k]==i:
                gran=reduce_arg['gransize'][k]
        utils.model_size(model)
        utils.save_checkpoint(model,args,'Reduced_Model{}.pt'.format(i))
        utils.get_evalues(model,valid_loader,reduce_arg['eval_pct'],reduce_arg['eval_top'])
        utils.reduce_precision(model,gran,args.quantize)
        utils.zero_evalues(model)
        _,_,correct=test(0,test_loader,'Test')
        best_correct=0
        for j in range(reduce_arg['epochs']):
            train(j, Log=True)
            _,_,correct=test(j,test_loader,'Test')
            utils.save_checkpoint(model,args,'Checkpoint{}.pt'.format(i))
            if correct>best_correct:
                utils.save_checkpoint(model,args,'Best{}.pt'.format(i))
                best_correct=correct
            if j in reduce_arg['lr_steps']:
                #optimizer.param_groups[0]['lr']=args.lr/10
                #args.lr=args.lr/10
                args.lr=args.lr
            print('Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))
        _,_,correct=test(i,test_loader,'Test')

elif args.mode == 'evaluate':
    folder,modelName=os.path.split(args.checkpoint)
    modelName=modelName.replace('.pt','')
    pred_test,target_test,correct=test(0,test_loader,'Test')
    outdict=utils.plotsizemobilenet(model,'weight')
    io.savemat(os.path.join(folder,modelName+'_WeightQuant.mat'),outdict)
    pickle.dump(outdict,open(os.path.join(folder,modelName+'_WeightQuant.pkl'),'wb'))
    plt.savefig(os.path.join(folder,modelName+'_WeightQuant.pdf'))
    print('Plot saved at: '+ os.path.join(folder, modelName+'_WeightQuant.pdf'))
    outdict=utils.plotsizemobilenet(model,'activation')
    pickle.dump(outdict,open(os.path.join(folder,modelName+'_ActQuant.pkl'),'wb'))
    io.savemat(os.path.join(folder,modelName+'_ActQuant.mat'),outdict)
    plt.savefig(os.path.join(folder,modelName+'_ActQuant.pdf'))
    print('Activation plot saved at: '+ os.path.join(folder, modelName+'_ActQuant.pdf'))

elif args.mode == 'ActquantApt':
    utils.quantize_model(model, args)
    for i in range(1,16):
        print('------------------Act Bits{}------------------'.format(i))
        utils.sync_clip(model, args,i)
        pred_test,target_test,correct=test(0,test_loader,'Test')

elif args.mode == 'train':
    best_correct=0
    log_file_test  = open(os.path.join(args.log_dir,'LearningCurveTest.txt'), "w")
    log_file_train  = open(os.path.join(args.log_dir,'LearningCurveTrain.txt'), "w")
    log_file_test.write('epoch\ttest-loss\tcorrect\ttest-size')
    log_file_train.write('epoch\ttrain-loss\tcorrect\ttrain-size')
    while (args.epoch < args.epochs+1):
        t = time.time()
        optimizer.param_groups[0]['lr']=utils.lr_scheduler(args)
        print('Learning rate is {}'.format(optimizer.param_groups[0]['lr']))
        train(args.epoch, Log=True)
        _,_,correct=test(args.epoch,test_loader,'Test')
        log_file_test.flush()
        log_file_train.flush()
        utils.save_checkpoint(model,args,'Checkpoint.pt')
        if correct>best_correct:
            best_correct=correct
            utils.save_checkpoint(model,args,'Best.pt')
            print('New Best Accuracy: {:0.2f}'.format(100. * float(best_correct) / len(test_loader.dataset)))
        args.epoch=args.epoch+1;
        # do stuff
        elapsed = time.time() - t
        print('Time Taken: {}'.format(elapsed))
    print('Best Accuracy : {:0.2f}%'.format(100. * float(best_correct) / len(test_loader.dataset)))

else:
    print('Choose a Mode')
