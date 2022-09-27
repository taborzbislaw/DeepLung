import os
import numpy as np
import glob
import argparse
import pickle
import json
import sys
import math

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

        key, img, target, feat = dum['id'], dum['image'], dum['label'],dum['features']

        if self.transform is not None:
            img = self.transform(img)

        return key, img, target, feat, self.weights[index]

    def __len__(self):
        return self.len

if __name__ == '__main__':

    fold = '0'
    if len(sys.argv)>1:
        fold = sys.argv[1]

    patchesPath = './patchesTs_fold' + fold + '/'
    patchesList = np.asarray([f for f in sorted(glob.glob(patchesPath + '*.npy')) if 'statistics' not in os.path.basename(f) and 'weights' not in os.path.basename(f)])

    statsFileName = './patchesTr_fold' + fold + '/statistics.npy'
    stats = np.load(statsFileName,allow_pickle=True).item()

    batchSize = 1
    numWorkers = 8

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((stats['meanSignalValue']), (stats['stdSignalValue'])),
    ])

    test_weights = [1 for _ in range(len(patchesList))]
    testSet = nodData(patchesList,test_weights,transform=transform_test,maxDiam = stats['maxDiam'])

    test_gen = torch.utils.data.DataLoader(testSet, batch_size=batchSize, num_workers=numWorkers,shuffle=False)

    #######################################################################
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True

    networks = []
    checkpoints = glob.glob('./clsModels/testFold_' + fold + 'trainFold_*_model_best.model')

    for checkpoint in checkpoints:
        network = densenet_small()
        saved_model = torch.load(checkpoint, map_location=torch.device('cpu'))
        network.load_state_dict(saved_model['state_dict'])
        network.eval()
        if use_cuda:
            network.cuda()
        networks.append(network)


    preds = {}
    for batch_idx, (keys, imgs, targets, _, _) in enumerate(test_gen):
        key = keys[0]
        print(batch_idx, len(test_gen))
        #print(batch_idx,inputs.shape,targets.shape,feat.shape,end = ' ')
        if use_cuda:
            imgs, targets = imgs.cuda(), targets.cuda()

        mean = 0
        for network in networks:
            mean += network(imgs).detach().cpu().numpy()[0][0]/len(networks)

        mean = 1/(1 + math.exp(-mean))
        #print(mean)

        if key not in preds.keys():
            preds[key] = [mean]
        else:
            preds[key].append(mean)


    gtPath = './PROCESSED/'    
    for key in preds.keys():
        status = 1
        fname = gtPath + key + '_lesions.npy'
        dum = np.load(fname,allow_pickle = True)
        if dum.shape[0] == 0:
            status = 0

        print(key,status,dum.shape[0],preds[key])
        f = open('clsPredictions_fold' + fold + '.txt','at')
        print(key,status,dum.shape[0],preds[key],file=f)
        f.close()

