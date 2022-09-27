import torch
from torch import nn
from layers import *

class Net(nn.Module):
    def __init__(self,config):
        super(Net, self).__init__()
        # The first few layers consumes the most memory, so use simple convolution to save memory.
        # Call these layers preBlock, i.e., before the residual blocks of later layers.
        self.config = config
        self.preBlock = nn.Sequential(
            nn.Conv3d(1, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True),
            nn.Conv3d(24, 24, kernel_size = 3, padding = 1),
            nn.BatchNorm3d(24),
            nn.ReLU(inplace = True))
        
        # 3 poolings, each pooling downsamples the feature map by a factor 2.
        # 3 groups of blocks. The first block of each group has one pooling.
        num_blocks_forw = [2,2,3,3]
        num_blocks_back = [3,3]
        self.featureNum_forw = [24,32,64,64,64]
        self.featureNum_back =    [128,64,64]
        for i in range(len(num_blocks_forw)):
            blocks = []
            for j in range(num_blocks_forw[i]):
                if j == 0:
                    blocks.append(PostRes(self.featureNum_forw[i], self.featureNum_forw[i+1]))
                else:
                    blocks.append(PostRes(self.featureNum_forw[i+1], self.featureNum_forw[i+1]))
            setattr(self, 'forw' + str(i + 1), nn.Sequential(*blocks))

        for i in range(len(num_blocks_back)):
            blocks = []
            for j in range(num_blocks_back[i]):
                if j == 0:
                    if i==0:
                        addition = 3
                    else:
                        addition = 0
                    blocks.append(PostRes(self.featureNum_back[i+1]+self.featureNum_forw[i+2]+addition, self.featureNum_back[i]))
                else:
                    blocks.append(PostRes(self.featureNum_back[i], self.featureNum_back[i]))
            setattr(self, 'back' + str(i + 2), nn.Sequential(*blocks))

        self.maxpool1 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool2 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool3 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.maxpool4 = nn.MaxPool3d(kernel_size=2,stride=2,return_indices =True)
        self.unmaxpool1 = nn.MaxUnpool3d(kernel_size=2,stride=2)
        self.unmaxpool2 = nn.MaxUnpool3d(kernel_size=2,stride=2)

        self.path1 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.path2 = nn.Sequential(
            nn.ConvTranspose3d(64, 64, kernel_size = 2, stride = 2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace = True))
        self.drop = nn.Dropout3d(p = 0.5, inplace = False)
        self.output = nn.Sequential(nn.Conv3d(self.featureNum_back[0], 64, kernel_size = 1),
                                    nn.ReLU(),
                                    #nn.Dropout3d(p = 0.3),
                                   nn.Conv3d(64, 5 * len(self.config['anchors']), kernel_size = 1))

    def forward(self, x, coord):
        # print('x, coord',x.shape, coord.shape)#torch.Size([BS, 1, 96, 96, 96]) torch.Size([BS, 3, 24, 24, 24])
        out = self.preBlock(x)#torch.Size([BS, 24, 96, 96, 96])
        
        out_pool,indices0 = self.maxpool1(out)#torch.Size([BS, 24, 48, 48, 48])
        out1 = self.forw1(out_pool)#torch.Size([BS, 32, 48, 48, 48])
        out1_pool,indices1 = self.maxpool2(out1)#torch.Size([BS, 32, 24, 24, 24])
        out2 = self.forw2(out1_pool)#torch.Size([BS, 64, 24, 24, 24])
        #out2 = self.drop(out2)
        out2_pool,indices2 = self.maxpool3(out2)#torch.Size([BS, 64, 12, 12, 12])
        out3 = self.forw3(out2_pool)#torch.Size([BS, 64, 12, 12, 12])
        out3_pool,indices3 = self.maxpool4(out3)#torch.Size([BS, 64, 6, 6, 6])
        out4 = self.forw4(out3_pool)#torch.Size([BS, 64, 6, 6, 6])
        #out4 = self.drop(out4)
        # print('out',out_pool.shape,out1.shape,out1_pool.shape,out2.shape,out2_pool.shape,out3.shape,out3_pool.shape,out4.shape)
        rev3 = self.path1(out4)#torch.Size([7, 64, 12, 12, 12])
        comb3 = self.back3(torch.cat((rev3, out3), 1))#torch.Size([7, 64, 12, 12, 12])
        #comb3 = self.drop(comb3)
        rev2 = self.path2(comb3)#torch.Size([7, 64, 24, 24, 24])
        
        comb2 = self.back2(torch.cat((rev2, out2,coord), 1))#64+64
        comb2 = self.drop(comb2)#torch.Size([7, 128, 24, 24, 24])
        out = self.output(comb2)#torch.Size([7, 15, 24, 24, 24])
        size = out.size()
        # print('out',out.shape)
        out = out.view(out.size(0), out.size(1), -1)#torch.Size([7, 15, 13824])
        # print('out',out.shape)
        #out = out.transpose(1, 4).transpose(1, 2).transpose(2, 3).contiguous()
        out = out.transpose(1, 2).contiguous().view(size[0], size[2], size[3], size[4], len(self.config['anchors']), 5)
        #out = out.view(-1, 5)
        
        return out#torch.Size([7, 24, 24, 24, 3, 5])

    
def get_model(config):
    net = Net(config)
    loss = Loss(config['num_hard'])
    get_pbb = GetPBB(config)
    return net, loss, get_pbb


