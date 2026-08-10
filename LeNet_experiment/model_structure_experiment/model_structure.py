"""LeNet 模型结构消融配置。

当前项目的真实基线是 CIFAR-10 RGB 输入、卷积通道 3 -> 16 -> 32。
本文件完全独立于上级目录的 model.py，不会影响已有超参数实验。
"""

from copy import deepcopy

import torch
import torch.nn as nn


MODEL_CONFIGS = {
    'baseline_current': {
        'category': '基线',
        'modification': '当前模型：Conv通道3→16→32，k=5，MaxPool，ReLU',
        'model_kwargs': {},
    },
    'channel_reduced_4_8': {
        'category': '卷积通道消融',
        'modification': 'Conv通道由3→16→32减少为3→4→8',
        'model_kwargs': {'conv_channels': (4, 8)},
    },
    'channel_classic_6_16': {
        'category': '卷积通道消融',
        'modification': '采用经典LeNet通道规模，RGB输入下为3→6→16',
        'model_kwargs': {'conv_channels': (6, 16)},
    },
    'depth_add_conv': {
        'category': '网络深度消融',
        'modification': '在第二次池化后增加Conv3(k=3,p=1)+MaxPool',
        'model_kwargs': {'extra_conv': True},
    },
    'kernel_size_3': {
        'category': '卷积核大小消融',
        'modification': '两个基础卷积层kernel_size由5改为3',
        'model_kwargs': {'kernel_size': 3},
    },
    'kernel_size_7': {
        'category': '卷积核大小消融',
        'modification': '两个基础卷积层kernel_size由5改为7',
        'model_kwargs': {'kernel_size': 7},
    },
    'pooling_avg': {
        'category': '池化方式消融',
        'modification': 'MaxPool2d改为AvgPool2d',
        'model_kwargs': {'pool_type': 'avg'},
    },
    'pooling_none': {
        'category': '池化方式消融',
        'modification': '取消两次Pooling，Linear输入维度自动调整',
        'model_kwargs': {'pool_type': 'none'},
    },
    'activation_sigmoid': {
        'category': '激活函数消融',
        'modification': 'ReLU改为Sigmoid',
        'model_kwargs': {'activation': 'sigmoid'},
    },
    'activation_tanh': {
        'category': '激活函数消融',
        'modification': 'ReLU改为Tanh',
        'model_kwargs': {'activation': 'tanh'},
    },
    'activation_leaky_relu': {
        'category': '激活函数消融',
        'modification': 'ReLU改为LeakyReLU(negative_slope=0.01)',
        'model_kwargs': {'activation': 'leaky_relu'},
    },
    'batch_norm': {
        'category': 'Batch Normalization消融',
        'modification': '每个卷积层后加入BatchNorm2d',
        'model_kwargs': {'use_batch_norm': True},
    },
    'dropout_0_5': {
        'category': 'Dropout消融',
        'modification': '两个FC隐藏层激活后加入Dropout(p=0.5)',
        'model_kwargs': {'dropout': 0.5},
    },
}


def _make_activation(name):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(negative_slope=0.01),
    }
    if name not in activations:
        raise ValueError('不支持的激活函数：{}'.format(name))
    return activations[name]


def _make_pool(pool_type):
    pools = {
        'max': nn.MaxPool2d(kernel_size=2, stride=2),
        'avg': nn.AvgPool2d(kernel_size=2, stride=2),
        'none': nn.Identity(),
    }
    if pool_type not in pools:
        raise ValueError('不支持的池化方式：{}'.format(pool_type))
    return pools[pool_type]


class LeNetStructure(nn.Module):
    """支持单因素结构切换、自动推断Linear输入维度的LeNet。"""

    def __init__(
            self,
            input_channels=3,
            input_size=32,
            num_classes=10,
            conv_channels=(16, 32),
            kernel_size=5,
            pool_type='max',
            activation='relu',
            extra_conv=False,
            use_batch_norm=False,
            dropout=0.0):
        super().__init__()

        if len(conv_channels) != 2:
            raise ValueError('conv_channels必须包含两个通道数。')
        if not 0.0 <= dropout < 1.0:
            raise ValueError('dropout必须位于[0, 1)范围内。')

        channel1, channel2 = conv_channels
        self.input_channels = input_channels
        self.input_size = input_size
        self.extra_conv_enabled = extra_conv

        self.conv1 = nn.Conv2d(input_channels, channel1, kernel_size)
        self.bn1 = (
            nn.BatchNorm2d(channel1) if use_batch_norm else nn.Identity()
        )
        self.pool1 = _make_pool(pool_type)

        self.conv2 = nn.Conv2d(channel1, channel2, kernel_size)
        self.bn2 = (
            nn.BatchNorm2d(channel2) if use_batch_norm else nn.Identity()
        )
        self.pool2 = _make_pool(pool_type)

        if extra_conv:
            # 5x5基础特征图经过k=3、padding=1后尺寸不变，可继续2倍下采样。
            self.conv3 = nn.Conv2d(
                channel2, channel2, kernel_size=3, padding=1
            )
            self.bn3 = (
                nn.BatchNorm2d(channel2)
                if use_batch_norm else nn.Identity()
            )
            self.pool3 = _make_pool(pool_type)

        self.activation = _make_activation(activation)

        flattened_features = self._infer_flattened_features()
        self.fc1 = nn.Linear(flattened_features, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def _forward_features(self, x):
        x = self.pool1(self.activation(self.bn1(self.conv1(x))))
        x = self.pool2(self.activation(self.bn2(self.conv2(x))))
        if self.extra_conv_enabled:
            x = self.pool3(self.activation(self.bn3(self.conv3(x))))
        return x

    def _infer_flattened_features(self):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy_input = torch.zeros(
                1, self.input_channels, self.input_size, self.input_size
            )
            features = self._forward_features(dummy_input)
        self.train(was_training)
        return features.flatten(start_dim=1).shape[1]

    def forward(self, x):
        x = self._forward_features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.dropout(self.activation(self.fc2(x)))
        return self.fc3(x)


def create_model(config_name):
    """按配置名创建全新的模型实例。"""
    if config_name not in MODEL_CONFIGS:
        raise KeyError('未知模型配置：{}'.format(config_name))
    config = MODEL_CONFIGS[config_name]
    return LeNetStructure(**deepcopy(config['model_kwargs']))


def count_parameters(model):
    """统计可训练参数量。"""
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)
