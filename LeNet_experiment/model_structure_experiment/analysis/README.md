# Parameter Efficiency Analysis

## 模块用途

本目录用于对已经完成的LeNet模型结构消融结果进行“参数量—测试准确率—训练成本”后处理分析。脚本不会导入训练脚本、不会重新训练模型，也不会修改原始Excel或模型权重。

## 文件说明

```text
analysis/
├── parameter_accuracy_analysis.py   # 独立分析脚本
├── parameter_accuracy_results.csv   # 标准化数据及效率指标
├── parameter_efficiency_report.md   # 自动生成的分析报告
├── parameter_accuracy_tradeoff.png  # 参数量—准确率论文风格散点图
├── parameter_efficiency_bubble.png  # 训练时间气泡图
└── README.md                         # 本说明
```

## 数据输入

默认自动读取：

```text
../results/model_structure_results.xlsx
```

也可以通过 `--input` 指定 `.xlsx`、`.csv` 或 `.json`。脚本支持常见字段别名，并统一整理为：

| 字段 | 含义 |
| --- | --- |
| `model_name` | 模型或实验名称 |
| `parameters` | 可训练参数量 |
| `accuracy` | 最佳测试准确率，CSV中以百分数保存 |
| `training_time` | 训练时间（秒），允许缺失 |
| `category` | 结构消融类别 |

对于当前Excel，字段映射为：

```text
Experiment        -> model_name
Parameters        -> parameters
Best Accuracy     -> accuracy
Training Time (s) -> training_time
Category          -> category
```

新增模型记录写入原结果文件的 `summary` 工作表后，重新运行脚本即可自动纳入表格、Pareto分析和图片。

## 效率指标

以Baseline为基准：

```text
参数增长比例 = (parameter_new - parameter_baseline) / parameter_baseline
Accuracy变化 = accuracy_new - accuracy_baseline
Accuracy提升比例 = Accuracy变化 / accuracy_baseline
参数效率 = Accuracy提升比例 / 参数增长比例
```

参数量不变时参数效率记为N/A。对于减参且准确率也下降的模型，参数效率比值可能为正，因此报告只在“参数增加且准确率提高”的候选中评选最高提升效率，并另外使用Pareto前沿判断整体trade-off。

## 运行方法

在本目录直接运行：

```powershell
python parameter_accuracy_analysis.py
```

指定其他数据文件或Baseline：

```powershell
python parameter_accuracy_analysis.py --input other_results.csv --baseline baseline_current
```

查看参数说明：

```powershell
python parameter_accuracy_analysis.py --help
```

脚本依赖 `numpy`、`matplotlib` 和 `openpyxl`，与当前LeNet实验环境一致。
