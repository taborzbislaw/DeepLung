
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
from split_combine import SplitComb

def test(split_comber, data, target, coord, nzhw, net, get_pbb, thresh, isfeat, n_per_run):

        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        data = data[0][0]
        coord = coord[0][0]

        splitlist = range(0, len(data) + 1, n_per_run)
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
#            print(input.shape)
            inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())

        output = np.concatenate(outputlist, 0)
        output = split_comber.combine(output, nzhw=nzhw)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]

        if isfeat:
            return feature_selected,pbb,lbb
        else:
            #print(type(pbb),type(lbb))
            return pbb,lbb


def test1(data_loader, net, get_pbb, save_dir, config):

    save_dir = os.path.join(save_dir, 'bbox')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    split_comber = data_loader.dataset.split_comber
    for i_name, (data, target, coord, nzhw) in enumerate(data_loader):
        target = [np.asarray(t, np.float32) for t in target]
        lbb = target[0]
        nzhw = nzhw[0]
        name = os.path.basename(data_loader.dataset.filenames[i_name])[0:4]  # .split('-')[0]  wentao change
        data = data[0][0]
        coord = coord[0][0]
        
        
        isfeat = False
        # isfeat = True
        if 'output_feature' in config:
            if config['output_feature']:
                isfeat = True
        n_per_run = config['n_test']
        print('data.size', data.size(), n_per_run)
        splitlist = range(0, len(data) + 1, n_per_run)
        print('splitlist', splitlist)
        if splitlist[-1] != len(data):
            splitlist.append(len(data))
        outputlist = []
        featurelist = []

        for i in range(len(splitlist) - 1):
            input = Variable(data[splitlist[i]:splitlist[i + 1]]).cuda()
            inputcoord = Variable(coord[splitlist[i]:splitlist[i + 1]]).cuda()
            # print('input',input.shape)
            if isfeat:
                output, feature = net(input, inputcoord)
                featurelist.append(feature.data.cpu().numpy())
            else:
                output = net(input, inputcoord)
            outputlist.append(output.data.cpu().numpy())

        output = np.concatenate(outputlist, 0)
        # print('NET-output',output.shape)
        output = split_comber.combine(output, nzhw=nzhw)
        # print('COMBINE-output',output.shape)
        if isfeat:
            feature = np.concatenate(featurelist, 0).transpose([0, 2, 3, 4, 1])[:, :, :, :, :, np.newaxis]
            feature = split_comber.combine(feature, sidelen)[..., 0]

        thresh = config['testthresh']  # -8 #-3
        pbb, mask = get_pbb(output, thresh, ismask=True)
        if isfeat:
            feature_selected = feature[mask[0], mask[1], mask[2]]
            np.save(os.path.join(save_dir, name + '_feature.npy'), feature_selected)
        np.save(os.path.join(save_dir, name + '_pbb.npy'), pbb)
        np.save(os.path.join(save_dir, name + '_lbb.npy'), lbb)
        

if __name__ == "__main__":

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', required=True, help='json file with the configuration of the data needed to perform augmentation')
    argparser.add_argument('--testFold', required=True, help='test fold')
    args = argparser.parse_args()

    config_file_name = args.config
    test_fold = int(args.testFold)

    with open(config_file_name, 'r') as config_file:
        config = json.load(config_file)

    fnames = sorted(glob.glob(config['processed_data_dir'] + '*_img.npy'))

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

#######################################################################

#    margin = 16
#    sidelen = 128
    margin = 8
    sidelen = 80
    
    split_comber = SplitComb(sidelen, config['max_stride'], config['stride'], margin, config['pad_value'])

    test_dataset = data.DataBowl3Detector(config['processed_data_dir'],test_keys,config,phase = 'test',split_comber=split_comber)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['workers'],
        collate_fn=data.collate,
        pin_memory=False)

    trainVal_dataset = data.DataBowl3Detector(config['processed_data_dir'],trainVal_keys,config,phase = 'test',split_comber=split_comber)
    trainVal_loader = DataLoader(
        trainVal_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['workers'],
        collate_fn=data.collate,
        pin_memory=False)
    
    device = 'cuda'
    nets = []
    get_pbbs = []
    checkpoints = glob.glob(config['save_dir'] + 'best_test_' + str(test_fold) + '_train_*_.ckpt')
    for item in checkpoints:
        net, _ , get_pbb = get_model(config)
        checkpoint = torch.load(item)
        net.load_state_dict(checkpoint['state_dict'])
        net.eval()
        net = net.to(device)
        nets.append(net)
        get_pbbs.append(get_pbb)

    save_dir = config['save_preds']

    save_dir_Ts = save_dir + 'Ts_fold' + str(test_fold)
    if not os.path.exists(save_dir_Ts):
        os.makedirs(save_dir_Ts)

    save_dir_Tr = save_dir + 'Tr_fold' + str(test_fold)
    if not os.path.exists(save_dir_Tr):
        os.makedirs(save_dir_Tr)

    isfeat = False
    if 'output_feature' in config:
        if config['output_feature']:
            isfeat = True
    thresh = config['testthresh'] 
    n_per_run = config['n_test']

    for i_name, (data, target, coord, nzhw) in enumerate(test_loader):
        name = os.path.basename(test_loader.dataset.filenames[i_name])[0:4]
        if os.path.isfile(os.path.join(save_dir_Ts, name + '_lbb.npy'))==True:
            continue
        features = []
        pbbs = []
        lbbs = []
        for  net,get_pbb in zip(nets, get_pbbs):
            out = test(test_loader.dataset.split_comber, data, target, coord, nzhw, net, get_pbb, thresh, isfeat, n_per_run)
            if isfeat:
                features.append(out[0])
                pbbs.append(out[1])
                lbbs.append(out[2])
            else:
                pbbs.append(out[0])
                lbbs.append(out[1])
        np.save(os.path.join(save_dir_Ts, name + '_pbb.npy'), pbbs)
        np.save(os.path.join(save_dir_Ts, name + '_lbb.npy'), lbbs[0])  #all are the same

        if len(features) > 0:
            np.save(os.path.join(save_dir_Ts, name + '_feature.npy'), features)
        
    for i_name, (data, target, coord, nzhw) in enumerate(trainVal_loader):
        name = os.path.basename(trainVal_loader.dataset.filenames[i_name])[0:4]
        if os.path.isfile(os.path.join(save_dir_Tr, name + '_lbb.npy'))==True:
            continue
        features = []
        pbbs = []
        lbbs = []
        for  net,get_pbb in zip(nets, get_pbbs):
            out = test(trainVal_loader.dataset.split_comber, data, target, coord, nzhw, net, get_pbb, thresh, isfeat, n_per_run)
            if isfeat:
                features.append(out[0])
                pbbs.append(out[1])
                lbbs.append(out[2])
            else:
                pbbs.append(out[0])
                lbbs.append(out[1])

        np.save(os.path.join(save_dir_Tr, name + '_pbb.npy'), pbbs)
        np.save(os.path.join(save_dir_Tr, name + '_lbb.npy'), lbbs[0])  #all are the same

        if len(features) > 0:
            np.save(os.path.join(save_dir_Tr, name + '_feature.npy'), features)


