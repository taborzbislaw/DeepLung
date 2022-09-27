import os
import numpy as np
import glob
import argparse
import pickle
import json
import sys

"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import WeightedRandomSampler


class nodData(data.Dataset):
    def __init__(self, fileList, weights,transform=None,maxDiam = -1):
        self.transform = transform
        self.dataset = fileList
        self.weights = weights
        #np.random.shuffle(self.dataset)
        self.maxDiam = maxDiam
        self.len = len(self.dataset)

    def __getitem__(self, index):
        dum = np.load(self.dataset[index],allow_pickle=True).item()

        if self.maxDiam > 0:
            dum['features'][-1] /= self.maxDiam

        img, target, feat, weight = dum['image'], dum['label'],dum['features'], self.weights[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target, feat, weight

    def __len__(self):
        return self.len



if __name__ == '__main__':

    fold = 0
    maxEpochs = 5

    if len(sys.argv)>1:
        fold = int(sys.argv[1])

    if len(sys.argv)>2:
        maxEpochs = int(sys.argv[2])

    SEED = 42
    patchesPath = './patchesTr/'
    savemodelpath = './clsModels/'

    logName = savemodelpath + 'log_' + str(fold) + '.txt'
    saveFreq = 10

    batchSize = 16
    numWorkers = 8
    lr = 0.01
    neptime = 0.3
    CROPSIZE = 17
    gbtdepth = 2

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

    transform_train = None
    transform_val = None

    train_weights = torch.from_numpy(train_weights)
    train_weigths = train_weights.double()
    trainSampler = WeightedRandomSampler(train_weights, len(train_weights),replacement=True)

    val_weights = torch.from_numpy(val_weights)
    val_weigths = val_weights.double()
    valSampler = WeightedRandomSampler(val_weights, len(val_weights),replacement = True)

    trainSet = nodData(train_keys,train_weigths,transform=transform_train,maxDiam = stats['maxDiam'])
    valSet = nodData(val_keys,val_weigths,transform=transform_val,maxDiam = stats['maxDiam'])
    
    trainLoader = torch.utils.data.DataLoader(trainSet, batch_size=batchSize, num_workers=numWorkers,sampler=trainSampler)
    valLoader = torch.utils.data.DataLoader(valSet, batch_size=batchSize, num_workers=numWorkers,sampler=valSampler)

    num = 0
    pos = 0
    for batch_idx, (inputs, targets, feat, weight) in enumerate(trainLoader):
        print(targets[0],weight[0])
        num += targets.shape[0]
        pos += torch.sum(targets).numpy()
        if batch_idx > 200:
            break

    print(pos/num)



    """
if __name__ == '__main__':

    fold = 0

    if len(sys.argv)>1:
        fold = int(sys.argv[1])

    patchesPath = './patchesTr_fold' + str(fold) + '/'

    patchesList = np.asarray([f for f in sorted(glob.glob(patchesPath + '*.npy')) if 'statistics' not in os.path.basename(f) and 'weights' not in os.path.basename(f)])

    pos = 0
    neg = 0
    labels = []
    for num, patchName in enumerate(patchesList):

        if num%10 == 0:
            print(num,flush=True)

        dum = np.load(patchName,allow_pickle=True).item()
        if dum['label'] == 0:
            neg += 1
        else:
            pos += 1
        labels.append(dum['label'])

    wPos = 1/pos
    wNeg = 1/neg
    weights = [wNeg if l==0 else wPos for l in labels]

    np.save(patchesPath + 'weights.npy',weights)

    print(pos,neg)
    

