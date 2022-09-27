import re
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import get_norm


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate,
                 norm_type='Unknown'):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', get_norm(norm_type, num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', get_norm(norm_type, bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)  # noqa
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, norm_type='Unknown'):  # noqa
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate,
                                norm_type=norm_type)  # noqa
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features,
                 norm_type='Unknown'):
        super(_Transition, self).__init__()
        self.add_module('norm', get_norm(norm_type, num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,  # noqa
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool3d(kernel_size=2, stride=2))


class DenseNet3d(nn.Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer  # noqa
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), 
                 norm_type='Unknown', num_init_features=64, bn_size=4, drop_rate=0, num_classes=1):  # noqa

        super(DenseNet3d, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),  # noqa
            ('norm0', get_norm(norm_type, num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=num_features, norm_type=norm_type,  # noqa
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)  # noqa
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                    norm_type=norm_type)  # noqa
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', get_norm(norm_type, num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        self.num_features = num_features

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.InstanceNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        ################################
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        ###############################
        return x

def densenet_small(**kwargs):
    """Densenet-121 model from"""
    norm_type = 'BatchNorm'
    model = DenseNet3d(num_init_features=64, growth_rate=32, block_config=(6, 12, 24),  # noqa
                     norm_type=norm_type, **kwargs)
    return model


def densenet121(config, **kwargs):
    """Densenet-121 model from"""
    model = DenseNet3d(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),  # noqa
                     norm_type=config['norm_type'], **kwargs)
    return model


def densenet169(config, **kwargs):
    """Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa
    """
    model = DenseNet3d(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),  # noqa
                     norm_type=config['norm_type'], **kwargs)
    return model


def densenet201(config, **kwargs):
    """Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa
    """
    model = DenseNet3d(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),  # noqa
                     norm_type=config['norm_type'], **kwargs)
    return model


def densenet161(config, **kwargs):
    """Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_  # noqa
    """
    model = DenseNet3d(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),  # noqa
                     norm_type=config['norm_type'], **kwargs)
    return model
