import os
import numpy as np
import glob
import argparse
import pickle
import json
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler

from time import time
from torch.cuda.amp import GradScaler, autocast
from torchinfo import summary

import transforms as transforms
from densenet3d import densenet_small

def maybe_to_torch(d):
    if 'list' in type(d).__name__:
        d = [maybe_to_torch(i) if 'Tensor' not in type(i).__name__ else i for i in d]
    elif 'Tensor' not in type(d).__name__:
        d = torch.from_numpy(d).float()
    return d


class nodData(data.Dataset):
    def __init__(self, fileList, weights, transform=None,maxDiam = -1):
        self.transform = transform
        self.dataset = fileList
        self.weights = weights
        self.maxDiam = maxDiam
        self.len = len(self.dataset)

    def __getitem__(self, index):
        dum = np.load(self.dataset[index],allow_pickle=True).item()

        if self.maxDiam > 0:
            dum['features'][-1] /= self.maxDiam

        img, target, feat = dum['image'], dum['label'],dum['features']

        if self.transform is not None:
            img = self.transform(img)

        return img, target, feat, self.weights[index]

    def __len__(self):
        return self.len

def generator(dataLoader):
    while True:
        for inputs, targets, feats, weights in dataLoader:
            yield inputs,targets,feats,weights

def run_iteration(tr_gen, network, optimizer, loss, amp_grad_scaler, fp16=True, do_backprop=True, CEloss_weight=1.0):

    data,trueLabels,_,_ = next(tr_gen)

    trueLabels = trueLabels.cpu().detach().numpy()
    trueLabels = np.asarray(trueLabels, dtype=np.int64)

    data = maybe_to_torch(data)
    trueLabels = maybe_to_torch(trueLabels)

    if torch.cuda.is_available():
        data = data.cuda(0, non_blocking=True)
        trueLabels = trueLabels.cuda(0, non_blocking=True)

    optimizer.zero_grad()
    if fp16:
        with autocast():
            output = network(data)

            del data
            l = loss(output, trueLabels.unsqueeze(1))

        if do_backprop:
            amp_grad_scaler.scale(l).backward()
            amp_grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
            amp_grad_scaler.step(optimizer)
            amp_grad_scaler.update()
    else:
        output = network(data)
        del data
        l = loss(output,trueLabels.unsqueeze(1))

        if do_backprop:
            l.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 12)
            optimizer.step()

    return l.detach().cpu().numpy()


def save_checkpoint(fname, network, optimizer, amp_grad_scaler, epoch, lr, all_tr_losses, all_val_losses, best_epoch,
                    best_loss):
    state_dict = network.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    print("saving checkpoint...", flush=True)

    save_this = {
        'epoch': epoch + 1,
        'learning_rate': lr,
        'state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'amp_grad_scaler': amp_grad_scaler.state_dict(),
        'plot_stuff': (all_tr_losses, all_val_losses),
        'best_stuff': (best_epoch, best_loss)}

    torch.save(save_this, fname)
    print("saving done", flush=True)


if __name__ == '__main__':

    test_fold = 0
    fold = 0
    checkpoint = None

    if len(sys.argv)>1:
        test_fold = int(sys.argv[1])

    if len(sys.argv)>2:
        fold = int(sys.argv[2])

    if len(sys.argv) > 3:
        checkpoint = sys.argv[3]
    
    SEED = 42
    patchesPath = './patchesTr_fold' + str(test_fold) + '/'

    batchSize = 8
    numWorkers = 8

    patchesList = np.asarray([f for f in sorted(glob.glob(patchesPath + '*.npy')) if 'statistics' not in os.path.basename(f) and 'weights' not in os.path.basename(f)])

    weightsFileName = patchesPath + 'weights.npy'
    assert os.path.isfile(weightsFileName)==True,'no file with patches weights'

    samples_weight = np.asarray(np.load(weightsFileName,allow_pickle=True))

    numAll = len(patchesList)
    numFolds = 5
    foldSize = numAll//numFolds

    np.random.seed(SEED)
    ids = np.arange(0,len(patchesList))
    np.random.shuffle(ids)

    patchesList = patchesList[ids]
    samples_weight = samples_weight[ids]

    val_keys = [patchesList[i] for i in range(fold*foldSize,min((fold+1)*foldSize,len(patchesList)))]
    train_keys = [i for i in patchesList if i not in val_keys]

    val_weights = np.asarray([samples_weight[i] for i in range(fold*foldSize,min((fold+1)*foldSize,len(patchesList)))])
    train_weights = np.asarray([samples_weight[i] for i in range(0,len(patchesList)) if i not in range(fold*foldSize,min((fold+1)*foldSize,len(patchesList)))])

    statsFileName = patchesPath + 'statistics.npy'
    assert os.path.isfile(statsFileName)==True,'no file with patches stats'

    stats = np.load(statsFileName,allow_pickle=True).item()

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomYFlip(),
       transforms.RandomZFlip(),
        transforms.ZeroOut(4),
        transforms.ToTensor(),
        transforms.Normalize((stats['meanSignalValue']), (stats['stdSignalValue'])),  # need to cal mean and std, revise norm func
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((stats['meanSignalValue']), (stats['stdSignalValue'])),
    ])

    train_weights = torch.from_numpy(train_weights)
    train_weigths = train_weights.double()
    trainSampler = WeightedRandomSampler(train_weights, len(train_weights),replacement=True)

    val_weights = torch.from_numpy(val_weights)
    val_weigths = val_weights.double()
    valSampler = WeightedRandomSampler(val_weights, len(val_weights),replacement = True)

    trainSet = nodData(train_keys,train_weights,transform=transform_train,maxDiam = stats['maxDiam'])
    valSet = nodData(val_keys,val_weights,transform=transform_val,maxDiam = stats['maxDiam'])

    tr_gen = generator(torch.utils.data.DataLoader(trainSet, batch_size=batchSize, num_workers=numWorkers,sampler=trainSampler))
    val_gen = generator(torch.utils.data.DataLoader(valSet, batch_size=batchSize, num_workers=numWorkers,sampler=valSampler))

    #######################################################################

    network = densenet_small()

    if torch.cuda.is_available():
        network.cuda()

    initial_lr = 0.001
    momentum = 0.99
    nesterov = True
    weight_decay = 3e-5
    optimizer = torch.optim.SGD(network.parameters(), initial_lr, weight_decay=weight_decay,
                                momentum=momentum, nesterov=nesterov)

    ########################################################################
    #########                TRAINING CONFIG                 ###############
    ########################################################################

    # na wyjściu warstwa gęsta BEZ AKTYWACJI!!!
    loss = nn.BCEWithLogitsLoss()

    fp16 = True
    amp_grad_scaler = GradScaler()

    numOfEpochs = 1500
    tr_batches_per_epoch = 250
    val_batches_per_epoch = 50
    checkpoint_frequency = 10
    outputDir = './clsModels/'
    log_file = outputDir + 'log_' + str(test_fold) + '_' + str(fold) + '.txt'

    all_tr_losses = []
    all_val_losses = []

    startEpoch = 0

    bestValLoss = 1e30
    bestEpoch = 0

    if checkpoint != None:
        print('loading model from', checkpoint, flush=True)
        saved_model = torch.load(checkpoint, map_location=torch.device('cpu'))

        startEpoch = saved_model['epoch']
        initial_lr = saved_model['learning_rate']
        all_tr_losses, all_val_losses = saved_model['plot_stuff']
        bestEpoch, bestValLoss = saved_model['best_stuff']

        amp_grad_scaler.load_state_dict(saved_model['amp_grad_scaler'])
        optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        network.load_state_dict(saved_model['state_dict'])

        print('model loaded', flush=True)


    for epoch in range(startEpoch, numOfEpochs):

        network.train()

        print('epoch ', epoch, network.training, flush=True)

        lr =  initial_lr
        optimizer.param_groups[0]['lr'] = lr

        epoch_start_time = time()
        train_losses_epoch = []

        for batchNo in range(tr_batches_per_epoch):
            if batchNo % 10 == 0:
                print('#', end='', flush=True)
            l = run_iteration(tr_gen, network, optimizer, loss, amp_grad_scaler, fp16=True, do_backprop=True)
            train_losses_epoch.append(l)

        print('\n', flush=True)
        all_tr_losses.append(np.mean(train_losses_epoch))

        with torch.no_grad():
            network.eval()
            val_losses = []
            for _ in range(val_batches_per_epoch):
                print('>', end='', flush=True)
                l = run_iteration(val_gen, network, optimizer, loss, amp_grad_scaler, fp16=True, do_backprop=False, )
                val_losses.append(l)

            print('\n', flush=True)

            all_val_losses.append(np.mean(val_losses))

        epoch_end_time = time()

        print("epoch: ", epoch, ", training loss: ", all_tr_losses[-1], ", validation loss: ", all_val_losses[-1],
              ',  this epoch took: ', epoch_end_time - epoch_start_time, 's', flush=True)

        f = open(log_file, 'at')
        print("epoch: ", epoch, ", training loss: ", all_tr_losses[-1], ", validation loss: ", all_val_losses[-1],
              ', this epoch took: ', epoch_end_time - epoch_start_time, 's', file=f)
        f.close()

        if all_val_losses[-1] < bestValLoss:
            bestValLoss = all_val_losses[-1]
            bestEpoch = epoch
            fname = outputDir + '/testFold_' + str(test_fold) + 'trainFold_' + str(fold) + '_model_best.model'
            save_checkpoint(fname, network, optimizer, amp_grad_scaler, epoch, lr, all_tr_losses, all_val_losses, bestEpoch,
                            bestValLoss)

        if epoch % checkpoint_frequency == checkpoint_frequency - 1:
            fname = outputDir + '/testFold_' + str(test_fold) + 'trainFold_' + str(fold) + '_model_latest.model'
            save_checkpoint(fname, network, optimizer, amp_grad_scaler, epoch, lr, all_tr_losses, all_val_losses, bestEpoch,
                            bestValLoss)

    fname = outputDir + '/testFold_' + str(test_fold) + 'trainFold_' + str(fold) + '_model_final.model'
    save_checkpoint(fname, network, optimizer, amp_grad_scaler, epoch, lr, all_tr_losses, all_val_losses, bestEpoch,
                    bestValLoss)

