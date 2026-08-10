# LeNet CIFAR-10 消融实验项目

## 项目简介

本项目以 LeNet 和 CIFAR-10 为基础，系统研究训练超参数与网络结构变化对模型收敛、测试准确率、参数规模和训练成本的影响。项目包含三部分：

1. **超参数消融实验**：学习率、训练轮数和优化器。
2. **模型结构消融实验**：通道数、深度、卷积核、Pooling、激活函数、BatchNorm和Dropout。
3. **参数效率分析**：分析可训练参数量、最佳测试准确率和训练时间之间的trade-off。

所有实验结果均已保存，可以直接阅读报告和查看图表，不需要重新训练模型。

## 项目结构

```text
LeNet_experiment/
├── README.md                         # 项目总览（本文件）
├── data/                             # 共享CIFAR-10数据集
├── hyperparameter_experiment/        # 超参数消融实验
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── visualize_results.py
│   ├── README.md                     # 超参数实验详细报告
│   └── results/
│       ├── hyperparameter_results.xlsx
│       ├── models/
│       └── figures/
└── model_structure_experiment/       # 模型结构消融实验
    ├── model_structure.py
    ├── train_structure.py
    ├── visualize_structure_results.py
    ├── README.md                     # 结构实验详细报告
    ├── results/
    │   ├── model_structure_results.xlsx
    │   ├── summary.md
    │   ├── models/
    │   └── figures/
    └── analysis/                     # 参数量—准确率效率分析
        ├── parameter_accuracy_analysis.py
        ├── parameter_accuracy_results.csv
        ├── parameter_efficiency_report.md
        ├── parameter_accuracy_tradeoff.png
        ├── parameter_efficiency_bubble.png
        └── README.md
```

## 实验环境与统一设置

| 项目 | 设置 |
| --- | --- |
| 数据集 | CIFAR-10 |
| 输入 | RGB图像，3×32×32 |
| 训练集 | 50,000张图像 |
| 测试指标 | 固定5,000张测试图像上的accuracy |
| Batch Size | 36 |
| 损失函数 | CrossEntropyLoss |
| 数据归一化 | mean=0.5，std=0.5 |
| 主要环境 | PyTorch、torchvision、openpyxl、matplotlib、numpy |

> 当前 `test_accuracy` 使用测试集DataLoader的第一个5,000张图像，并非完整10,000张CIFAR-10测试集。比较结果时应保持这一评价方法一致。

## 一、超参数消融实验

详细报告：[hyperparameter_experiment/README.md](hyperparameter_experiment/README.md)

### 实验变量

| 类别 | 对照设置 |
| --- | --- |
| 学习率 | 0.001、0.002、0.003、0.005 |
| 训练轮数 | 5、10、15、20轮 |
| 优化器 | Adam、SGD、RMSprop、Adagrad |

### 核心结果

- Adam、学习率0.001、训练约10轮获得本组最高测试准确率 **70.38%**。
- 训练超过10轮后，训练损失继续下降，但测试准确率进入平台并出现回落，说明继续训练主要增强训练集拟合。
- 学习率从0.001增大到0.005后，5轮最终准确率由67.36%降至46.58%，说明当前Adam设置对过大学习率较敏感。
- 相同学习率0.001和5轮预算下，Adam与RMSprop明显优于Adagrad和无动量SGD；该结论不代表SGD在单独调节学习率和动量后仍然较差。

![训练轮数对照](hyperparameter_experiment/results/figures/训练轮数对照.png)

## 二、模型结构消融实验

详细报告：[model_structure_experiment/README.md](model_structure_experiment/README.md)

### 当前结构基线

```text
输入3通道
→ Conv1: 3→16, kernel=5
→ MaxPool
→ Conv2: 16→32, kernel=5
→ MaxPool
→ FC: 800→120→84→10
→ ReLU，无BatchNorm，无Dropout
```

基线参数量为121,182，最终准确率为69.78%，最佳准确率为70.38%。结构实验统一使用Adam、学习率0.001、训练10轮和随机种子42。

### 核心结果

- **BatchNorm效果最好**：仅增加96个参数，最终准确率提高到 **73.06%**。
- **通道数影响容量**：通道由16→32减少为6→16和4→8后，最终准确率分别降至64.54%和59.50%。
- **增加深度不一定增大模型**：增加Conv3和第三次Pooling后，参数量降至49,790，最佳准确率仍达到69.48%，是较好的轻量化折中。
- **kernel=5更均衡**：kernel=3训练损失更低但泛化未提升；kernel=7因无padding下空间压缩过快，最终准确率下降到65.30%。
- **Pooling不可简单取消**：无Pooling模型达到2,237,022个参数、训练时间360.2秒，但最终准确率只有62.66%，表现出明显过拟合。
- **ReLU与LeakyReLU表现接近**，Sigmoid收敛最慢；Dropout(p=0.5)对当前小模型正则化过强。

![模型结构消融总体结果](model_structure_experiment/results/figures/01_总体指标对比.png)

## 三、参数量—准确率效率分析

- 模块说明：[model_structure_experiment/analysis/README.md](model_structure_experiment/analysis/README.md)
- 完整报告：[parameter_efficiency_report.md](model_structure_experiment/analysis/parameter_efficiency_report.md)
- 标准化数据：[parameter_accuracy_results.csv](model_structure_experiment/analysis/parameter_accuracy_results.csv)

### 核心结果

- 13个结构配置中，`log10(参数量)` 与最佳准确率的相关系数仅为 **0.078**，参数增加与性能提升几乎没有稳定线性关系。
- BatchNorm参数只增加0.08%，最佳准确率提高2.68个百分点，是参数效率最高的增参方案。
- Add-Conv3以约41%的基线参数保留69.48%的最佳准确率，是紧凑模型中的有效选择。
- No-Pooling参数增长1746%，训练时间最长且准确率下降，属于计算成本增加但收益为负的结构。
- Pareto前沿模型为 Channel-4-8、Add-Conv3、Baseline和BatchNorm。

![Parameter Accuracy Trade-off](model_structure_experiment/analysis/parameter_accuracy_tradeoff.png)

![Parameter Efficiency Bubble](model_structure_experiment/analysis/parameter_efficiency_bubble.png)

## 综合结论

实验结果说明，模型性能不能通过单独增加训练轮数、学习率或参数量来保证：

1. **优化稳定性比盲目增大更新步长更重要。** Adam配合0.001学习率表现稳定，学习率过大明显损害收敛。
2. **训练时间存在合理上限。** 约10轮后准确率趋于稳定，继续训练会扩大训练拟合与测试性能之间的差距。
3. **结构质量比参数规模更重要。** BatchNorm以极小参数代价获得最高准确率，无Pooling却在参数量暴涨后性能下降。
4. **下采样与特征容量需要平衡。** 通道太少会形成表示瓶颈，完全取消Pooling则造成参数冗余和过拟合。
5. **当前推荐方向**：以 `3→16→32、kernel=5、MaxPool、ReLU` 为基础，加入BatchNorm；训练使用Adam、学习率0.001，并通过验证集在约10轮附近选择最佳检查点。

## 阅读顺序建议

如果希望系统学习本项目，建议按以下顺序阅读：

1. 阅读超参数报告，理解学习率、epoch和优化器如何影响梯度更新与收敛。
2. 阅读结构报告，学习特征图尺寸、参数量计算、过拟合和结构设计。
3. 阅读参数效率报告，理解Accuracy、模型规模和训练成本之间的Pareto trade-off。
4. 对照Excel原始记录和PNG曲线，验证报告中的每一个结论。

## 运行与复现

### 超参数实验

```powershell
cd hyperparameter_experiment
python train.py
python visualize_results.py
```

### 模型结构实验

```powershell
cd model_structure_experiment
python train_structure.py --list
python train_structure.py
python visualize_structure_results.py
```

### 参数效率分析（不重新训练）

```powershell
cd model_structure_experiment\analysis
python parameter_accuracy_analysis.py
```

重新运行训练前请关闭正在使用结果文件的Excel或WPS。训练脚本会重写同名实验记录和模型权重；如果只需要查看现有结果，无需执行训练命令。

## 实验结论的适用范围

- 每个配置主要使用单个固定随机种子，尚未通过重复实验计算均值、标准差和置信区间。
- 不同优化器共享学习率0.001，因此优化器对比反映的是固定预算下的表现，不是各优化器充分调参后的上限。
- 某些结构修改会同时改变特征图尺寸、全连接输入和参数量，结论对应完整配置变化。
- 训练时间与本机硬件和系统负载有关，适合比较同一环境内的相对成本。
