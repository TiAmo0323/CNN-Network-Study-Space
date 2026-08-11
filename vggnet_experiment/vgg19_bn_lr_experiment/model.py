"""独立的 VGG19 + BatchNorm 模型定义。"""

import torch
import torch.nn as nn


VGG19_CONFIG = [
    64,
    64,
    "M",
    128,
    128,
    "M",
    256,
    256,
    256,
    256,
    "M",
    512,
    512,
    512,
    512,
    "M",
    512,
    512,
    512,
    512,
    "M",
]


def make_features():
    layers = []
    in_channels = 3
    for value in VGG19_CONFIG:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            convolution = nn.Conv2d(in_channels, value, kernel_size=3, padding=1)
            layers.extend(
                [convolution, nn.BatchNorm2d(value), nn.ReLU(inplace=True)]
            )
            in_channels = value
    return nn.Sequential(*layers)


class VGG19BN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = make_features()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )
        self.initialize_weights()

    def forward(self, inputs):
        outputs = self.features(inputs)
        outputs = self.avgpool(outputs)
        outputs = torch.flatten(outputs, start_dim=1)
        return self.classifier(outputs)

    def initialize_weights(self):
        # 与已有 VGG/BN 实验保持完全相同的初始化方法。
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)


def vgg19_bn(num_classes=5):
    return VGG19BN(num_classes=num_classes)
