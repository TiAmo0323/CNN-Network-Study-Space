from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image

from model import LeNet # 导入模型


def main():
    script_dir = Path(__file__).resolve().parent
    # 数据预处理：将图片转换为Tensor，并进行归一化处理
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    net = LeNet() # 创建模型
    model_path = script_dir / 'results' / 'models' / 'Lenet.pth'
    net.load_state_dict(torch.load(model_path)) # 加载模型参数

    im = Image.open(script_dir / '1.jpg') # 打开图片
    im = transform(im)  # [C, H, W]   数据处理，转为tensor
    im = torch.unsqueeze(im, dim=0)  # [N, C, H, W] 增加batch维度，变为[1,3,32,32]

    with torch.no_grad(): # 预测阶段：关闭梯度
        outputs = net(im)
        predict = torch.max(outputs, dim=1)[1].numpy() # 得到预测类别
    print(classes[int(predict)]) # 打印对应类别


if __name__ == '__main__':
    main()
# 只有当这个Python文件被直接执行时，才执行main()函数；
# 如果这个文件被其他文件导入，则不会执行main()函数
