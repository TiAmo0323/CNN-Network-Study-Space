# VGG Network Experiments

这是一个基于PyTorch的VGG图像分类实验项目，使用五分类花卉数据集研究以下问题：

1. VGG11、VGG13、VGG16、VGG19在相同训练策略下的深度差异。
2. Batch Normalization对不同深度VGG训练稳定性与性能的影响。
3. Batch Normalization加入后，learning rate对VGG19优化过程的影响。

仓库保留实验代码、配置、逐epoch指标、汇总CSV、曲线和Markdown报告。由于GitHub单文件大小限制，数据集、模型权重和优化器断点不上传，需要在本地重新下载数据并训练生成。

## 主要实验结论

- 无BN深度消融中，VGG11取得最高验证准确率86.54%；VGG19在原训练策略下没有有效收敛。
- 加入BN后，VGG19恢复有效训练能力，说明BN能够缓解深层VGG的优化困难；但BN没有统一提高所有浅层VGG的准确率。
- VGG19-BN学习率实验中，`learning rate=0.00005`表现最好，最佳留出集准确率为77.75%，并取得最低训练loss和测试loss。
- BN改善深层网络可训练性，但仍需针对具体模型、数据集、batch size和优化器调整学习率。

详细结论见：

- [VGG深度消融报告](model_type_experiment/EXPERIMENT.md)
- [BatchNorm稳定性分析](model_BN_experiment/EXPERIMENT_ANALYSIS.md)
- [VGG19-BN学习率实验报告](vgg19_bn_lr_experiment/EXPERIMENT_REPORT.md)

## 项目结构

```text
vggnet_experiment/
├── README.md                         # 项目总说明
├── requirements.txt                  # Python依赖
├── model.py                           # 支持VGG11/13/16/19及可选BN的模型定义
├── train.py                           # VGG11/13/16/19-BN训练入口
├── compare_bn_results.py              # Baseline与BN结果比较及绘图
├── predict.py                         # 单张图片预测示例
├── prepare_data.py                    # 固定随机种子划分train/val
├── class_indices.json                 # 花卉类别映射
├── data/                              # 本地数据，不上传GitHub
├── model_type_experiment/             # 无BN的VGG深度消融
│   ├── code_snapshot/                 # Baseline训练代码快照
│   ├── results/                       # CSV、配置、图表；权重不上传
│   └── EXPERIMENT.md                  # 实验设置、结果与结论
├── model_BN_experiment/               # BN对不同深度VGG的影响
│   ├── results/                       # BN组逐epoch结果
│   ├── comparison/                    # Baseline vs BN汇总与图表
│   ├── README.md
│   └── EXPERIMENT_ANALYSIS.md
└── vgg19_bn_lr_experiment/            # VGG19-BN学习率消融
    ├── model.py                       # 独立VGG19-BN模型
    ├── train.py                       # 多学习率训练与断点续训
    ├── analyze_results.py             # 汇总、绘图和Markdown报告生成
    ├── results/                       # 各学习率CSV和配置
    ├── outputs/                       # 四类对比曲线与summary.csv
    └── EXPERIMENT_REPORT.md
```

## 环境准备

推荐使用独立的Conda环境：

```bash
conda create -n DeepLearningStudy python=3.10 -y
conda activate DeepLearningStudy
pip install -r requirements.txt
```

项目训练时使用的主要环境为PyTorch 2.13.0、CUDA 12.6和NVIDIA RTX 4060 Laptop GPU。其他兼容版本也可运行；安装GPU版PyTorch时，建议根据本机CUDA环境参考PyTorch官方安装说明。

## 数据集来源与准备

本项目使用TensorFlow官方示例中的五分类花卉数据集，包含3670张图片：

- daisy
- dandelion
- roses
- sunflowers
- tulips

官方下载地址：

```text
https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
```

下载并准备数据：

```powershell
New-Item -ItemType Directory -Force data\flower_data
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz" `
  -OutFile "data\flower_photos.tgz"
tar -xzf data\flower_photos.tgz -C data\flower_data
python prepare_data.py
```

`prepare_data.py`使用固定随机种子0，将约90%图片划分到训练集、10%划分到留出集。期望目录结构为：

```text
data/flower_data/
├── flower_photos/     # 原始数据，3670张
├── train/             # 训练集，3306张
└── val/               # 固定留出集，364张
```

为避免误删数据，当前脚本检测到`train`或`val`已存在时会停止，不会自动覆盖。

## 实验一：VGG模型深度消融

实验比较无BN的VGG11、VGG13、VGG16和VGG19。原始代码与已生成结果位于`model_type_experiment`。

查看结果：

```text
model_type_experiment/results/summary.csv
model_type_experiment/results/comparison_curves.png
model_type_experiment/EXPERIMENT.md
```

如需从已有最后断点恢复：

```powershell
python model_type_experiment\code_snapshot\train.py --resume
```

GitHub版本不包含`best.pth`和`last_checkpoint.pth`。若没有本地断点，请指定一个新的输出目录重新训练。

## 实验二：BatchNorm与VGG深度

BN组将所有卷积块从：

```text
Conv2d -> ReLU
```

改为：

```text
Conv2d -> BatchNorm2d -> ReLU
```

训练四个BN模型：

```powershell
python train.py
```

从已有本地断点继续：

```powershell
python train.py --resume
```

重新生成Baseline与BN综合比较：

```powershell
python compare_bn_results.py
```

结果位于：

```text
model_BN_experiment/results/bn_group/
model_BN_experiment/comparison/
model_BN_experiment/EXPERIMENT_ANALYSIS.md
```

## 实验三：VGG19-BN学习率消融

该实验保持数据、增强、VGG19-BN结构、Adam、batch size、epoch和随机种子不变，比较：

```text
0.00005, 0.0001（已有BN结果）, 0.0002, 0.0003, 0.0005, 0.001
```

运行或恢复全部新学习率实验：

```powershell
python vgg19_bn_lr_experiment\train.py --resume
```

单独重新生成汇总、曲线和报告：

```powershell
python vgg19_bn_lr_experiment\analyze_results.py
```

输出包括：

- `history.csv`：每个epoch的train/test loss与accuracy。
- `outputs/summary.csv`：各学习率最佳指标。
- `train_loss_curve.png`
- `test_loss_curve.png`
- `train_accuracy_curve.png`
- `test_accuracy_curve.png`
- `EXPERIMENT_REPORT.md`

## 单张图片预测

`predict.py`演示了图片预处理、权重加载、softmax概率计算和类别映射。使用前需要：

1. 准备一张待预测图片并修改`img_path`。
2. 准备与模型结构匹配的本地`.pth`权重并修改`weights_path`。
3. 确认模型名称、是否使用BN以及类别数与权重一致。

模型权重未包含在GitHub仓库中。

## 指标与实验注意事项

- 每组实验保存逐epoch的Training Loss、Train Accuracy、Test/Validation Loss和Accuracy。
- 部分脚本为了与实验需求一致，将原`val`目录称为held-out test；但该集合还用于选择最佳epoch，因此严格来说仍属于验证集，而不是独立测试集。
- 当前结论基于单一随机种子和较小留出集。更严格的比较应使用多个随机种子，并报告均值和标准差。
- BatchNorm实验的batch size为8，较小batch可能导致运行均值和方差波动。

## GitHub中未包含的文件

以下内容通过`.gitignore`排除：

- `data/`及数据压缩包
- `*.pth`、`*.pt`、`*.ckpt`模型权重与训练断点
- `__pycache__/`和Python编译缓存
- 临时文件及IDE配置

CSV、JSON、PNG和Markdown实验结果均保留，可直接查看已有曲线和结论。
