import torch.nn as nn
import torch


class BasicBlock(nn.Module):# ResNet的残差块
# 两个3x3卷积层组成的残差块，适用于ResNet18/34
    expansion = 1# expansion表示输出通道数相对于输入通道数的扩展倍数，对于BasicBlock来说，输出通道数与输入通道数相同，因此expansion=1。

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x # 保存原始输入x，作为后面shortcut（跳跃连接）的分支
        if self.downsample is not None:
            identity = self.downsample(x)
        # downsample是一个卷积层+BN层的组合，用于将输入x的通道数和特征图尺寸调整为与残差块输出一致，以便进行相加操作。
        
        # 进入BasicBlock的主分支，进行两次卷积操作
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity # 关键：将输入的特征图与卷积后的特征图相加
        out = self.relu(out)

        return out


class Bottleneck(nn.Module): # 用于ResNet50/101/152的残差块
# Bottleneck残差块的设计灵感来自于“瓶颈”结构，通过1x1卷积层来减少通道数，从而降低计算量，同时保持网络的表达能力。
# 流程：输入->1x1卷积（降维）->3x3卷积->1x1卷积（升维）->输出
# 1x1卷积降维原因：减少计算量，降低参数数量，同时保留重要特征信息（因为3x3卷积计算量和channel有关）
# 第二层3x3卷积：提取空间特征（与普通CNN一样）
# 第三层1x1卷积升维：恢复通道数，保证输出特征图的维度与输入一致，以便进行残差连接（恢复channel数的原因是为了保证残差连接的维度一致性，便于后续的相加操作）
# 总结：先压缩，再计算，再扩展（就像一个瓶颈一样），这种设计使得网络在保持较高性能的同时，减少了计算资源的消耗。
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4# expansion表示输出通道数相对于输入通道数的扩展倍数，对于Bottleneck来说，输出通道数是输入通道数的4倍，因此expansion=4。

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels(squeeze:挤压)
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):# 主网络：把BasicBlock/Bottlenck这些小模块组合成完整的ResNet网络

    def __init__(self,
                 block, # block表示使用的残差块类型，可以是BasicBlock或Bottleneck
                 blocks_num, # 表示每个layer有多少个block组成的列表，例如[3, 4, 6, 3]表示layer1有3个block，layer2有4个block，layer3有6个block，layer4有3个block
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        # 第一层卷积Conv1（ResNet的输入部分）
        # 输入通道数为3（RGB图像），输出通道数为64，卷积核大小为7x7，步距为2，padding为3
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)# 对每个batch的特征进行归一化，提高训练稳定性
        # 卷积输出可能：很大、很小、分布变化；BatchNorm帮助：保持数据分布稳定

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet的核心部分：由4个残差层（layer1, layer2, layer3, layer4）组成，每个残差层由多个block组成
        # 创建残差层（layer1, layer2, layer3, layer4），每个残差层由多个block组成
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # AdaptiveAvgPool2d: 自适应平均池化层，将特征图的空间尺寸缩小为指定的输出尺寸（1x1），不管输入特征图的大小是多少，输出都是固定大小
        # AdaptiveAvgPool2d的作用：将不同尺寸的特征图统一为相同尺寸，便于后续全连接层处理
        # AdaptiveAvgPool 用来压缩空间维度，减少全连接参数，并让模型适应不同输入尺寸。
        # 假如输入特征图大小为512x4=2048，经过AdaptiveAvgPool2d后，变：[batch_size, 2048, 1, 1]，然后再通过flatten展平为[batch_size, 2048]，最后输入全连接层进行分类
        # 全连接层（fc）：将卷积层提取的特征映射到指定类别数的输出空间，用于分类任务
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):# 工程设计的关键--根据残差块的类型和数量，创建一个残差层
    # block: 残差块的类型，可以是BasicBlock或Bottleneck
    # channel: 残差块的输出通道数
    # block_num: 残差块的数量
    # stride: 残差块的步距，默认为1（决定这个layer是否进行下采样）
        downsample = None# 默认shortcut不需要改变
        if stride != 1 or self.in_channel != channel * block.expansion:
        # 如果步距不为1，说明需要进行下采样，或者输入通道数与输出通道数不匹配，也需要进行下采样
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))# 创建shortcut变换

        layers = []

        # 创建第一个残差块，可能需要进行下采样
        # 负责：改变channel和特征图尺寸（如果stride不为1）
        # 更新self.in_channel：将输入通道数调整为输出通道数的倍数（根据block.expansion）
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion # 更新当前输出channel，为后面的block做准备

        # 创建剩余的残差块，这些块不需要改变通道数和特征图尺寸（保持住）
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top: #分类头
            x = self.avgpool(x)# 作用：将特征图的空间尺寸压缩为1x1，得到每个通道的全局平均值
            x = torch.flatten(x, 1)# 将特征图展平为一维向量，方便输入全连接层进行分类
            x = self.fc(x)# 通过全连接层将特征映射到指定类别数的输出空间，得到最终的分类结果

        return x


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)

# LeNet验证了卷积神经网络通过卷积和池化提取图像特征的有效性；
# VGG通过堆叠多个3×3卷积构建更深网络，在增加表达能力的同时控制参数量；
# ResNet进一步引入残差连接，通过shortcut缓解深层网络优化困难，使非常深的网络能够稳定训练。