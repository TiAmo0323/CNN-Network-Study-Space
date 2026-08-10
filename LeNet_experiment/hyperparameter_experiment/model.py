import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5) #Conv2d卷积层处理空间结构
        # Conv2d(in_channels, out_channels, kernel_size)
        # Conv2d：卷积层，作用是从图片中提取特征
        # in_channels表示输入图片的通道数——3：RGB图片还有R，G，B三个通道
        # out_channels表示输出图片的通道数——16：卷积后输出的图片有16个通道（也就是卷积核数量）
        # 16个卷积核（filter1, filter2, ..., filter16）提取16种不同特征
        # （一个卷积核学习一个特征，最终得到16个特征图feature map）
        # kernel_size表示卷积核的大小（也就是卷积窗口大小），这里是5*5的卷积核
        # 卷积输出尺寸公式：output_size = (input_size + 2*padding - kernel_size) / stride + 1
        # stride表示卷积步长，这里默认是1，表示卷积核每次移动1个像素
        # padding表示卷积时在输入图片边缘补0的像素数，这里默认是0，表示不补0

        self.pool1 = nn.MaxPool2d(2, 2) # 池化：压缩特征图尺寸
        # MaxPool2d(kernel_size, stride)
        # kernel_size表示池化窗口大小，这里是2*2的池化窗口
        # stride表示步长，这里是2，表示池化窗口每次移动2个像素
        # 本例中，池化层的作用是将特征图的尺寸缩小一半（宽和高都缩小一半），从而减少计算量和参数量，同时保留主要特征信息
        # 池化层的输出尺寸计算公式：output_size = (input_size + 2*padding - kernel_size) / stride + 1
        # 本例中，输入图片尺寸为32*32，经过卷积层后输出尺寸为28*28，经过池化层后输出尺寸为14*14
        # 本例计算公式：output_size = (28 + 2*0 - 2) / 2 + 1 = 14

        self.conv2 = nn.Conv2d(16, 32, 5)
        # 第二个卷积层：输入16个通道（因为上一层输出16个通道），输出32个通道（学习更多特征），卷积核大小为5*5
        
        self.pool2 = nn.MaxPool2d(2, 2)# 再次池化
        # MaxPool（最大池化）是在一个局部区域内选择最大值作为输出。
        # 例如，输入特征图为：[[1, 2],[3, 4]]
        # 经过2x2的最大池化后，输出为4（即局部区域内的最大值）。
        # 池化层的作用是降低特征图的空间尺寸，从而减少参数量和计算量，同时保留主要特征信息。
        # 作用1：降低计算量
        # 作用2：增强模型的鲁棒性
        # 作用3：扩大感受野：后面的神经元看到更大的区域

        # 全连接层：将卷积层和池化层提取的特征映射到输出类别（处理向量）
        # 全连接层的参数（输入特征数量，输出特征数量）即输入多少维的向量，输出多少维的向量
        self.fc1 = nn.Linear(32*5*5, 120)
        # 32*5*5表示卷积层输出的特征图展平后的向量长度（32个通道，每个通道5*5的特征图）
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10) # 10表示10分类（输出10个类别的概率）

    def forward(self, x):
        x = F.relu(self.conv1(x))    # input(3, 32, 32) output(16, 28, 28)
        # ReLU给CNN引入非线性能力，使多个卷积层能够学习复杂特征
        # 执行顺序：输入图片->卷积提取特征->ReLU过滤激活值->输出

        x = self.pool1(x)            # output(16, 14, 14)
        x = F.relu(self.conv2(x))    # output(32, 10, 10)
        x = self.pool2(x)            # output(32, 5, 5)
        x = x.view(-1, 32*5*5)       # output(32*5*5)
        # 作用：把卷积输出的二维feature map展平为一维向量，作为全连接层的输入
        # CNN->全连接 的关键桥梁（view作用是改变Tensor的形状，类似于numpy中的reshape）
        # view()函数的参数：-1表示自动计算该维度的大小，32*5*5表示展平后的向量长度

        x = F.relu(self.fc1(x))      # output(120)
        x = F.relu(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
        # 最后一层不加ReLU是因为输出的不是中间特征，而是最终分类结果的logits(类别分数)
        # 需要保留正负信息交给损失函数处理
        return x


