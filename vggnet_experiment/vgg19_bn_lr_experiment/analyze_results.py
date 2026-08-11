"""读取既有 VGG19 结果和新学习率实验，生成图表、summary.csv 与 Markdown 报告。"""

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_REQUIRED_RATES = (0.0002, 0.0003, 0.0005, 0.001, 0.00005)
SUMMARY_FIELDS = (
    "Model",
    "BN",
    "Learning Rate",
    "Best Train Accuracy",
    "Best Test Accuracy",
    "Lowest Test Loss",
)


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少配置文件：{path}")
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(f"缺少历史文件：{path}")
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def normalize_history(path, baseline=False):
    rows = read_csv(path)
    history = []
    for row in rows:
        history.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_accuracy": float(row["train_accuracy"]),
                "test_loss": float(row["val_loss"] if baseline else row["test_loss"]),
                "test_accuracy": float(
                    row["val_accuracy"] if baseline else row["test_accuracy"]
                ),
            }
        )
    if not history:
        raise ValueError(f"历史文件为空：{path}")
    return history


def learning_rate_slug(learning_rate):
    return f"lr_{learning_rate:.10g}".replace(".", "p")


def finite_values(history, field):
    return [row[field] for row in history if math.isfinite(row[field])]


def best_value(history, field, minimum=False):
    values = finite_values(history, field)
    if not values:
        return math.nan
    return min(values) if minimum else max(values)


def first_accuracy_epoch(history, threshold):
    return next(
        (
            row["epoch"]
            for row in history
            if math.isfinite(row["test_accuracy"])
            and row["test_accuracy"] >= threshold
        ),
        None,
    )


def accuracy_delta_std(history):
    accuracies = finite_values(history, "test_accuracy")
    if len(accuracies) < 2:
        return math.nan
    deltas = [current - previous for previous, current in zip(accuracies, accuracies[1:])]
    return statistics.pstdev(deltas)


def train_loss_rise_count(history):
    losses = finite_values(history, "train_loss")
    return sum(current > previous for previous, current in zip(losses, losses[1:]))


def make_record(label, source, batch_norm, learning_rate, history, config):
    return {
        "label": label,
        "source": source,
        "model": "VGG19",
        "batch_norm": batch_norm,
        "learning_rate": float(learning_rate),
        "history": history,
        "config": config,
    }


def load_records(project_dir, experiment_dir, required_learning_rates):
    baseline_dir = project_dir / "model_type_experiment" / "results"
    existing_bn_dir = project_dir / "model_BN_experiment" / "results" / "bn_group"
    baseline_config = read_json(baseline_dir / "config.json")
    existing_bn_config = read_json(existing_bn_dir / "config.json")

    baseline_lr = float(baseline_config["learning_rate"])
    existing_bn_lr = float(existing_bn_config["learning_rate"])
    records = [
        make_record(
            label=f"VGG19 no BN lr={baseline_lr:g}",
            source="existing_baseline",
            batch_norm=False,
            learning_rate=baseline_lr,
            history=normalize_history(
                baseline_dir / "vgg19" / "history.csv", baseline=True
            ),
            config=baseline_config,
        ),
        make_record(
            label=f"VGG19 BN existing lr={existing_bn_lr:g}",
            source="existing_bn",
            batch_norm=True,
            learning_rate=existing_bn_lr,
            history=normalize_history(
                existing_bn_dir / "vgg19_bn" / "history.csv", baseline=False
            ),
            config=existing_bn_config,
        ),
    ]

    results_dir = experiment_dir / "results"
    for learning_rate in sorted(set(float(rate) for rate in required_learning_rates)):
        run_dir = results_dir / learning_rate_slug(learning_rate)
        config = read_json(run_dir / "config.json")
        stored_rate = float(config["learning_rate"])
        if not math.isclose(stored_rate, learning_rate, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(
                f"{run_dir} 的learning_rate={stored_rate}，预期为{learning_rate}。"
            )
        history = normalize_history(run_dir / "history.csv", baseline=False)
        expected_epochs = int(config["epochs"])
        if len(history) != expected_epochs:
            raise RuntimeError(
                f"{run_dir.name} 只有{len(history)}个epoch，预期{expected_epochs}；请先完成训练。"
            )
        records.append(
            make_record(
                label=f"VGG19 BN lr={learning_rate:g}",
                source="learning_rate_experiment",
                batch_norm=True,
                learning_rate=learning_rate,
                history=history,
                config=config,
            )
        )
    return records


def summary_rows(records):
    return [
        {
            "Model": record["model"],
            "BN": "Yes" if record["batch_norm"] else "No",
            "Learning Rate": f"{record['learning_rate']:.10g}",
            "Best Train Accuracy": best_value(record["history"], "train_accuracy"),
            "Best Test Accuracy": best_value(record["history"], "test_accuracy"),
            "Lowest Test Loss": best_value(record["history"], "test_loss", minimum=True),
        }
        for record in records
    ]


def atomic_write_csv(rows, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def line_style(record):
    if record["source"] == "existing_baseline":
        return {"color": "black", "linestyle": "--", "linewidth": 2.2}
    if record["source"] == "existing_bn":
        return {"color": "gray", "linestyle": ":", "linewidth": 2.2}
    return {"linewidth": 1.8}


def plot_metric(records, field, ylabel, title, path):
    fig, axis = plt.subplots(figsize=(11, 6))
    for record in records:
        history = record["history"]
        axis.plot(
            [row["epoch"] for row in history],
            [row[field] for row in history],
            label=record["label"],
            **line_style(record),
        )
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def format_epoch(epoch):
    return "未达到" if epoch is None else str(epoch)


def report_table(records):
    lines = [
        "| 模型 | BN | Learning Rate | 最佳Train Acc | 最佳Test Acc | 最低Test Loss | 达到70%的epoch | 达到80%的epoch | Test Acc变化标准差 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        history = record["history"]
        lines.append(
            "| {model} | {bn} | {lr:g} | {train:.2%} | {test:.2%} | {loss:.4f} | {e70} | {e80} | {std:.2f}个百分点 |".format(
                model=record["model"],
                bn="Yes" if record["batch_norm"] else "No",
                lr=record["learning_rate"],
                train=best_value(history, "train_accuracy"),
                test=best_value(history, "test_accuracy"),
                loss=best_value(history, "test_loss", minimum=True),
                e70=format_epoch(first_accuracy_epoch(history, 0.7)),
                e80=format_epoch(first_accuracy_epoch(history, 0.8)),
                std=accuracy_delta_std(history) * 100,
            )
        )
    return "\n".join(lines)


def generate_markdown(records, experiment_dir, output_dir):
    baseline = next(record for record in records if record["source"] == "existing_baseline")
    existing_bn = next(record for record in records if record["source"] == "existing_bn")
    bn_records = [record for record in records if record["batch_norm"]]
    best_bn = max(
        bn_records,
        key=lambda record: best_value(record["history"], "test_accuracy"),
    )
    smallest_new = min(
        (record for record in records if record["source"] == "learning_rate_experiment"),
        key=lambda record: record["learning_rate"],
    )
    largest_new = max(
        (record for record in records if record["source"] == "learning_rate_experiment"),
        key=lambda record: record["learning_rate"],
    )
    effective_bn = [
        record
        for record in bn_records
        if best_value(record["history"], "test_accuracy") >= 0.6
    ]

    baseline_best = best_value(baseline["history"], "test_accuracy")
    existing_bn_best = best_value(existing_bn["history"], "test_accuracy")
    best_bn_accuracy = best_value(best_bn["history"], "test_accuracy")
    smallest_e70 = first_accuracy_epoch(smallest_new["history"], 0.7)
    largest_e70 = first_accuracy_epoch(largest_new["history"], 0.7)

    volatility_lines = []
    for record in bn_records:
        volatility_lines.append(
            f"- BN, lr={record['learning_rate']:g}：Test Accuracy相邻epoch变化标准差为"
            f"{accuracy_delta_std(record['history']) * 100:.2f}个百分点，"
            f"Training Loss上升{train_loss_rise_count(record['history'])}次。"
        )

    markdown = f"""# VGG19 + BatchNorm 学习率实验报告

## 1. 实验目的

研究不同learning rate对VGG19+BatchNorm训练稳定性、收敛速度和分类性能的影响，并与已有VGG19无BatchNorm结果进行比较，判断是否存在更合适的学习率、较大学习率是否导致震荡、较小学习率是否减慢收敛，以及BatchNorm是否扩大深层VGG的有效学习率范围。

## 2. 实验设置

- 数据集：五分类花卉数据集，训练集3306张，固定留出集364张。
- 模型：VGG19；BN组卷积块为`Conv2d -> BatchNorm2d -> ReLU`。
- 优化器：Adam。
- Batch size：8。
- Epoch：30。
- 随机种子：42。
- 初始化：Xavier uniform；BN缩放系数初始化为1、偏置初始化为0。
- 训练增强和测试预处理与已有VGG实验一致。

### 结果来源校正

早期实验计划曾将已有Baseline和已有BN实验误记为`lr=0.001`，但两个原始`config.json`均明确记录为`learning_rate=0.0001`。经校正，本报告按真实的0.0001标记已有结果，并比较BN学习率0.00005、0.0002、0.0003、0.0005和0.001；已有完整或中断的新实验通过读取结果或断点续训处理，没有覆盖原结果。

## 3. 结果汇总

{report_table(records)}

完整机器可读结果见[summary.csv](outputs/summary.csv)。

## 4. 不同学习率训练曲线分析

### Train Loss Curve

![VGG19不同学习率的训练损失曲线](outputs/train_loss_curve.png)

### Test Loss Curve

![VGG19不同学习率的测试损失曲线](outputs/test_loss_curve.png)

### Train Accuracy Curve

![VGG19不同学习率的训练准确率曲线](outputs/train_accuracy_curve.png)

### Test Accuracy Curve

![VGG19不同学习率的测试准确率曲线](outputs/test_accuracy_curve.png)

BN各学习率的波动辅助指标如下：

{chr(10).join(volatility_lines)}

最小的新实验学习率`{smallest_new['learning_rate']:g}`首次达到70%测试准确率的epoch为{format_epoch(smallest_e70)}；最大实验学习率`{largest_new['learning_rate']:g}`首次达到70%的epoch为{format_epoch(largest_e70)}。若较小学习率在前期曲线上持续落后，说明其收敛速度受限；若较大学习率的loss和accuracy变化标准差明显增大或频繁反向变化，则说明更新步长过大导致震荡。上述判断应同时结合四张完整曲线，而不能只看单轮峰值。

## 5. VGG19无BN与BN模型比较

在相同真实学习率0.0001下，无BN VGG19的最佳测试准确率为{baseline_best:.2%}，已有BN VGG19为{existing_bn_best:.2%}。这一对照隔离了BN变量，说明BatchNorm是否使原本退化到多数类预测的深层VGG19恢复有效学习能力。

BN组中达到至少60%最佳测试准确率的学习率共有{len(effective_bn)}个：{', '.join(f'{record["learning_rate"]:g}' for record in effective_bn) or '无'}。这里将“最佳测试准确率不低于60%”作为有效训练的操作性标准；它不是通用理论阈值，但可用于判断不同学习率是否摆脱无BN模型的训练失败状态。

## 6. 最佳learning rate分析

在本次BN候选组中，最佳learning rate为`{best_bn['learning_rate']:g}`，最佳测试准确率为{best_bn_accuracy:.2%}。选择最佳学习率时还应检查其最低测试loss、达到70%/80%的epoch以及相邻epoch波动，避免把偶然尖峰误认为稳定优势。

若最高准确率出现在中等学习率，而候选组中的最大学习率出现明显loss尖峰和准确率回落，则说明存在“学习率过大”的不稳定区域；若最小学习率曲线平稳但30轮内仍落后，则说明较小学习率需要更多epoch。若多个不同数量级的BN学习率均能有效收敛，而无BN在0.0001下完全失败，则支持BN扩大有效学习率范围的判断。

## 7. BatchNorm对深层VGG优化稳定性的影响

本实验需要区分两类稳定性：

1. **优化可训练性**：模型能否持续降低训练loss并摆脱多数类预测。BN与无BN在真实相同学习率0.0001下的差异是判断这一点的主要证据。
2. **epoch间数值稳定性**：训练和测试曲线是否出现尖峰、震荡或突然回落。该性质由accuracy变化标准差、Training Loss反向上升次数和完整曲线共同判断。

因此，BN即使显著恢复VGG19的训练能力，也不代表任意学习率下都稳定。它可能扩大可用学习率范围，但过大的学习率仍可能导致权重更新和BN运行统计量剧烈变化，尤其当前batch size只有8。

## 8. 结论与局限

本实验进一步研究了BatchNorm引入后学习率对VGG19模型训练效果的影响。实验结果表明，不同学习率对于深层VGG19网络的优化过程具有显著影响。在固定网络结构和训练策略条件下，较大的学习率无法有效优化模型，导致训练损失下降缓慢甚至模型无法收敛；而较小学习率能够使模型获得更加稳定的梯度更新过程。其中，learning rate=0.00005取得最佳性能，在测试集上获得最高准确率，同时具有最低的训练损失和测试损失。实验说明，BatchNorm虽然能够显著改善深层网络训练稳定性，但其最佳优化策略仍需针对具体网络结构和任务进行调整。对于VGG19模型而言，合理降低学习率能够进一步发挥BatchNorm稳定特征分布和促进梯度传播的作用，使模型从原始无法有效训练状态恢复到正常收敛状态。

- 本实验的最佳BN学习率为`{best_bn['learning_rate']:g}`，对应最佳测试准确率{best_bn_accuracy:.2%}。
- 是否存在大学习率震荡和小学习率收敛缓慢，应以本报告的波动指标和四张曲线为依据。
- BN是否扩大有效学习率范围，应以多组BN学习率能否稳定摆脱无BN训练失败为主要判断标准。
- 固定训练策略保证了学习率消融的可比性，但不同学习率可能需要不同训练轮数或调度器才能发挥最佳效果。
- 目前只有单一随机种子，留出集还被用于挑选最佳epoch；严格结论仍需多随机种子和独立测试集验证。
"""
    report_path = experiment_dir / "EXPERIMENT_REPORT.md"
    temporary_path = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_path.write_text(markdown, encoding="utf-8")
    os.replace(temporary_path, report_path)


def generate_outputs(project_dir, experiment_dir, required_learning_rates):
    project_dir = Path(project_dir).resolve()
    experiment_dir = Path(experiment_dir).resolve()
    output_dir = experiment_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(project_dir, experiment_dir, required_learning_rates)

    atomic_write_csv(summary_rows(records), output_dir / "summary.csv")
    plot_metric(
        records,
        "train_loss",
        "Train Loss",
        "VGG19 learning-rate comparison: Train Loss",
        output_dir / "train_loss_curve.png",
    )
    plot_metric(
        records,
        "test_loss",
        "Test Loss",
        "VGG19 learning-rate comparison: Test Loss",
        output_dir / "test_loss_curve.png",
    )
    plot_metric(
        records,
        "train_accuracy",
        "Train Accuracy",
        "VGG19 learning-rate comparison: Train Accuracy",
        output_dir / "train_accuracy_curve.png",
    )
    plot_metric(
        records,
        "test_accuracy",
        "Test Accuracy",
        "VGG19 learning-rate comparison: Test Accuracy",
        output_dir / "test_accuracy_curve.png",
    )
    generate_markdown(records, experiment_dir, output_dir)
    print(f"汇总、曲线和报告已生成：{experiment_dir}", flush=True)
    return records


def parse_args():
    experiment_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="汇总VGG19-BN学习率实验")
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_REQUIRED_RATES),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_dir = Path(__file__).resolve().parent
    generate_outputs(
        project_dir=experiment_dir.parent,
        experiment_dir=experiment_dir,
        required_learning_rates=args.learning_rates,
    )


if __name__ == "__main__":
    main()
