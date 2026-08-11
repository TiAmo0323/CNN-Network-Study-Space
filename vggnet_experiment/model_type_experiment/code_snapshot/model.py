import torch.nn as nn
import torch
# VGG：通过重复堆叠多个3x3卷积层，让网络变得更深
# 用更多的小卷积层替代少量大卷积层，在减少参数的同时增加网络深度和非线性表达能力。


# official pretrain weights
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}# 这些的区别：卷积层数量不同

# VGG整体：输入图片->features(大量Conv+Pool，得到卷积特征)->flatten（得到一维特征）->classifier(全连接)->输出类别
class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        # 与标准 VGG 实现一致，确保进入分类器前的特征尺寸固定为 7x7。
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        # 把卷积部分整体保存：features=(Conv+ReLU+Pool)+...如此循环

        self.classifier = nn.Sequential(# 全连接分类部分：类似LeNet
        # 不自己手写全连接层原因：因为VGG的全连接部分是一个固定顺序的组合
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        # N x 512*7*7
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_features(cfg: list):
    layers = [] # 用普通list保存每一层的网络结构，最后用nn.Sequential把它们组合起来
    # 原因：此时网络层还没有构建完成，需要动态添加
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}# 数字表示卷积层的输出通道数，M表示MaxPool层
 # VGG创新之一：用一个列表描述网络结构，再自动生成网络
 # VGG的卷积层数量不同，导致参数量不同，性能也不同。VGG16和VGG19是最常用的两个版本。
 # VGG16和VGG19的区别：VGG16有13个卷积层，VGG19有16个卷积层。VGG11和VGG13是较小的版本，参数量更少，适合资源有限的场景。
 # VGG11和VGG13的区别：VGG11有8个卷积层，VGG13有10个卷积层。VGG11和VGG13的参数量更少，适合资源有限的场景。


def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)
    return model
