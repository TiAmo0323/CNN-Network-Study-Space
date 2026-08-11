# VGG19 + BatchNorm 学习率消融实验

该目录是完全独立的实验，不修改、不覆盖以下已有实验：

- `../model_type_experiment`
- `../model_BN_experiment`

## 学习率来源说明

已有无BN和已有BN实验的`config.json`均记录真实学习率为`0.0001`。经校正，程序会：

1. 只读加载无BN、lr=0.0001结果作为原始Baseline。
2. 只读加载BN、lr=0.0001已有结果。
3. 独立训练BN学习率：0.00005、0.0002、0.0003、0.0005、0.001。

这避免了把0.0001结果错误标记为0.001，也不会覆盖任何既有结果。

## 输出结构

```text
vgg19_bn_lr_experiment/
├── results/
│   ├── lr_0p0005/
│   ├── lr_0p001/
│   ├── lr_0p0002/
│   ├── lr_0p0003/
│   └── lr_0p00005/
├── outputs/
│   ├── summary.csv
│   ├── train_loss_curve.png
│   ├── test_loss_curve.png
│   ├── train_accuracy_curve.png
│   └── test_accuracy_curve.png
└── EXPERIMENT_REPORT.md
```

每个学习率目录中的`history.csv`严格包含：

```text
epoch,train_loss,train_accuracy,test_loss,test_accuracy
```

训练意外中断后可使用`--resume`继续。程序检测到已有结果时默认拒绝覆盖。
