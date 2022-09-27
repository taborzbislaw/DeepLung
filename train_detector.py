
import os
import glob
import time
import numpy as np
import json
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from layers import *
from res18 import *
import data

def get_lr(epoch,max_epochs,init_lr):
    if epoch <= max_epochs * 1/3: #0.5:
        lr = init_lr
    elif epoch <= max_epochs * 2/3: #0.8:
        lr = 0.1 * init_lr
    elif epoch <= max_epochs * 0.8:
        lr = 0.05 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr

def train(data_loader, net, loss, optimizer, lr):
    
    start_time = time.time()
    
    net.train()
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    metrics = []

    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda(), requires_grad=True)
        target = Variable(target.cuda())
        coord = Variable(coord.cuda(), requires_grad=True)
        #data = Variable(data, requires_grad=True)
        #target = Variable(target)
        #coord = Variable(coord, requires_grad=True)

        output = net(data, coord)
        loss_output = loss(output, target)
        optimizer.zero_grad()
        loss_output[0].backward()
        optimizer.step()

        loss_output[0] = loss_output[0].item()
        
        metrics.append([(l.detach().cpu() if hasattr(l, 'cpu') else l) for l in loss_output])

    end_time = time.time()
    return metrics, end_time - start_time 


def validate(data_loader, net, loss):
    start_time = time.time()
    
    net.eval()

    metrics = []
    for i, (data, target, coord) in enumerate(data_loader):
        data = Variable(data.cuda())
        target = Variable(target.cuda())
        coord = Variable(coord.cuda())
        #data = Variable(data)
        #target = Variable(target)
        #coord = Variable(coord)

        output = net(data, coord)
        loss_output = loss(output, target, train = False)
        loss_output[0] = loss_output[0].item()
        metrics.append([(l.detach().cpu() if hasattr(l, 'cpu') else l) for l in loss_output])    
    end_time = time.time()
    return metrics, end_time - start_time 


if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', required=True, help='json file with the configuration of the data needed to perform augmentation')
    argparser.add_argument('--testFold',required=True)
    argparser.add_argument('--trainFold',required=True)
    args = argparser.parse_args()

    config_file_name = args.config
    test_fold = int(args.testFold)
    train_fold = int(args.trainFold)

    with open(config_file_name, 'r') as config_file:
        config = json.load(config_file)

    fnames = sorted(glob.glob(config['processed_data_dir'] + '*_img.npy'))
    #keys = [os.path.basename(f)[:4] for f in fnames]

    f = open(config['code_file'],'rt')
    lines = f.readlines()
    f.close()
    keys = [lines[i].split()[2][4:8] for i in np.arange(0,len(lines),2)]

#######################################################################
#   selekcja w oparciu o test_fold i train_fold                      ##
#######################################################################
    SEED = 42
    numAll = len(keys)
    numFolds = 5
    testFoldSize = numAll//numFolds
    numTrainVal = numAll - testFoldSize
    trainFoldSize = numTrainVal//numFolds

    np.random.seed(SEED)
    np.random.shuffle(keys)

    test_keys = [keys[i] for i in range(test_fold*testFoldSize,min((test_fold+1)*testFoldSize,len(keys)))]
    trainVal_keys = [i for i in keys if i not in test_keys]

    val_keys = [trainVal_keys[i] for i in range(train_fold*trainFoldSize,min((train_fold+1)*trainFoldSize,len(trainVal_keys)))]
    tr_keys = [ i for i in trainVal_keys if i not in val_keys]

    #tr_keys = keys[0:2]
    #val_keys = keys[0:2]
    #test_keys = keys[0:2]
#######################################################################

    train_dataset = data.DataBowl3Detector(config['processed_data_dir'],tr_keys,config,phase = 'train')
    val_dataset = data.DataBowl3Detector(config['processed_data_dir'],val_keys,config,phase = 'val')

    train_loader = DataLoader(
        train_dataset,
        batch_size = config['batch_size'],
        shuffle = True,
        num_workers = config['workers'],
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size = config['batch_size'],
        shuffle = False,
        num_workers = config['workers'],
        pin_memory=True)

    net, loss, get_pbb = get_model(config)
    start_epoch = 0
    best_loss = 1e10

    if config['resume'] != 'None':
        checkpoint = torch.load(config['resume'])
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

    loss = loss.cuda()
    device = 'cuda'
    net = net.to(device)
    #cudnn.benchmark = True#False 
    #net = DataParallel(net).cuda()

    optimizer = torch.optim.SGD(
        net.parameters(),
        config['init_lr'],
        momentum = 0.9,
        weight_decay = config['weight_decay'])

    for epoch in range(start_epoch, config['max_epochs'] + 1):

        lr = get_lr(epoch,config['max_epochs'],config['init_lr'])
        
        metrics, elapsed_time = train(train_loader, net, loss, optimizer, lr)

        metrics = np.asarray(metrics, dtype=np.float32)
        print('Epoch %03d (lr %.5f)' % (epoch, lr))
        print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            elapsed_time))
        print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5])))
        f = open(f'{config["logfile"]}_test_{test_fold}_train_{train_fold}_.txt','at')
        print('Epoch %03d (lr %.5f)' % (epoch, lr),end = ' ',file = f)
        print('Train:      tpr %3.2f, tnr %3.2f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            elapsed_time),end = ' ',file = f)
        print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5])),file = f)
        f.close()
        
        metrics, elapsed_time = validate(val_loader, net, loss)

        metrics = np.asarray(metrics, np.float32)
        print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            elapsed_time))
        print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5])))
        f = open(f'{config["logfile"]}_test_{test_fold}_train_{train_fold}_.txt','at')
        print('Epoch %03d (lr %.5f)' % (epoch, lr),end = ' ',file = f)
        print('Validation: tpr %3.2f, tnr %3.8f, total pos %d, total neg %d, time %3.2f' % (
            100.0 * np.sum(metrics[:, 6]) / np.sum(metrics[:, 7]),
            100.0 * np.sum(metrics[:, 8]) / np.sum(metrics[:, 9]),
            np.sum(metrics[:, 7]),
            np.sum(metrics[:, 9]),
            elapsed_time),end = ' ',file = f)
        print('loss %2.4f, classify loss %2.4f, regress loss %2.4f, %2.4f, %2.4f, %2.4f' % (
            np.mean(metrics[:, 0]),
            np.mean(metrics[:, 1]),
            np.mean(metrics[:, 2]),
            np.mean(metrics[:, 3]),
            np.mean(metrics[:, 4]),
            np.mean(metrics[:, 5])),file = f)
        f.close()

        if epoch % config['save_freq'] == 0:            
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
                
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'config': config,
                'metrics': metrics,
                'best_loss': best_loss},
                os.path.join(config['save_dir'], f'latest_test_{test_fold}_train_{train_fold}_.ckpt'))

        if np.mean(metrics[:, 0]) < best_loss:
            best_loss = np.mean(metrics[:, 0])
            print('best_loss',best_loss)
            state_dict = net.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()
                
            torch.save({
                'epoch': epoch,
                'state_dict': state_dict,
                'config': config,
                'metrics': metrics,
                'best_loss': best_loss},
                os.path.join(config['save_dir'], f'best_test_{test_fold}_train_{train_fold}_.ckpt'))

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
        
    torch.save({
        'epoch': epoch,
        'state_dict': state_dict,
        'config': config,
        'metrics': metrics,
        'best_loss': best_loss},
        os.path.join(config['save_dir'], f'final_test_{test_fold}_train_{train_fold}_.ckpt'))

                
