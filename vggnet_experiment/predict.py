import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import vgg

# 整体流程：读取图片-图片预处理-增加batch维度-读取类别映射json-创建VGG模型
# -加载训练权重-模型预测-得到概率最高类别-输出类别名称

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0) # 增加batch维度，变成[N, C, H, W]，N=1
    # 如果dim = 1，则在第1维增加一个维度，原来的第1维变成第2维，原来的第2维变成第3维，以此类推

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f) # 读取映射类别

    # create model
    model = vgg(model_name="vgg16", num_classes=5).to(device)
    # load model weights
    weights_path = "./vgg16Net.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval() #关闭Dropout（随机丢弃神经元），BatchNorm（训练产生的方差与均值等）固定
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0) # 计算每个类别的概率（总和为1）
        predict_cla = torch.argmax(predict).numpy() # 取概率最大的类别索引

    # 把模型预测结果（类别+概率）转换成人能看懂的形式，并打印出来、显示在图片标题上。
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    # 生成预测结果字符串
    # class_indict[str(predict_cla)]：根据预测编号找到类别名称。
    # {:.3}：保留三位小数，.3表示保留三位小数，.3f表示保留三位小数并且是浮点数

    plt.title(print_res) # 设置图片标题。把字符串print_res显示在图片标题上
    for i in range(len(predict)):# 遍历所有类别打印概率
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)], #str(i)把数字转换成字符串
                                                  predict[i].numpy()))
    plt.show() # 显示图片


if __name__ == '__main__':
    main()
