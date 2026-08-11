# Batch Normalization 对不同深度 VGG 训练稳定性的影响

## 实验分组

- Baseline：已有 VGG11/13/16/19，卷积块为 `Conv2d -> ReLU`
- BN：VGG11/13/16/19-BN，卷积块为 `Conv2d -> BatchNorm2d -> ReLU`

除是否加入 Batch Normalization 外，数据划分、图像预处理、初始化方法、优化器、学习率、batch size、epoch、随机种子和混合精度设置均与 Baseline 保持一致。

数据目录中的 `val` 是现有固定留出集。为了对应本实验指标命名，BN 历史及综合报告将其记作 held-out test；它不是额外重新划分出的第三份数据。

## 自动生成的指标

- 每轮 Training Loss
- 每轮 Train Accuracy
- 每轮 Test Loss 和 Test Accuracy
- 最佳 Test Accuracy 及对应 epoch
- 首次达到 80% Test Accuracy 的 epoch；未达到记为 `NR`
- 最后 5 轮 Test Accuracy 的总体标准差，用于辅助观察训练稳定性
- 最终 Train/Test Accuracy gap
- 每个模型训练时间

训练全部 BN 模型后，程序会自动读取 `../model_type_experiment/results` 中的 Baseline 结果，并在本目录的 `comparison` 子目录生成：

- `comparison_summary.csv`
- `training_loss_curves.png`
- `test_accuracy_curves.png`
- `train_vs_test_accuracy.png`
- `convergence_speed.png`
