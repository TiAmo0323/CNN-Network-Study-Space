"""汇总并可视化无 BN Baseline 与 BN 组实验结果。"""

import argparse
import csv
import os
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODELS = ("vgg11", "vgg13", "vgg16", "vgg19")
SUMMARY_FIELDS = (
    "group",
    "model",
    "parameters",
    "best_test_accuracy",
    "best_epoch",
    "final_test_accuracy",
    "final_train_accuracy",
    "final_train_loss",
    "epoch_to_80_accuracy",
    "test_accuracy_std_last_5",
    "generalization_gap_final",
    "training_seconds",
)


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def normalized_history(path, group):
    if not path.is_file():
        raise FileNotFoundError(f"缺少实验历史：{path}")
    rows = read_csv(path)
    history = []
    for row in rows:
        test_loss = row["val_loss"] if group == "Baseline" else row["test_loss"]
        test_accuracy = row["val_accuracy"] if group == "Baseline" else row["test_accuracy"]
        history.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_accuracy": float(row["train_accuracy"]),
                "test_loss": float(test_loss),
                "test_accuracy": float(test_accuracy),
                "epoch_seconds": float(row["epoch_seconds"]),
            }
        )
    if not history:
        raise ValueError(f"实验历史为空：{path}")
    return history


def first_threshold_epoch(history, threshold):
    return next(
        (row["epoch"] for row in history if row["test_accuracy"] >= threshold),
        None,
    )


def summarize(group, model_name, history, parameters, training_seconds, threshold):
    best = max(history, key=lambda row: row["test_accuracy"])
    final = history[-1]
    threshold_epoch = first_threshold_epoch(history, threshold)
    last_accuracies = [row["test_accuracy"] for row in history[-5:]]
    return {
        "group": group,
        "model": model_name,
        "parameters": parameters,
        "best_test_accuracy": best["test_accuracy"],
        "best_epoch": best["epoch"],
        "final_test_accuracy": final["test_accuracy"],
        "final_train_accuracy": final["train_accuracy"],
        "final_train_loss": final["train_loss"],
        "epoch_to_80_accuracy": "" if threshold_epoch is None else threshold_epoch,
        "test_accuracy_std_last_5": statistics.pstdev(last_accuracies),
        "generalization_gap_final": final["train_accuracy"] - final["test_accuracy"],
        "training_seconds": training_seconds,
    }


def atomic_write_csv(rows, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def plot_training_loss(histories, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for axis, model_name in zip(axes.flat, MODELS):
        for group, style in (("Baseline", "-"), ("BN", "--")):
            history = histories[(group, model_name)]
            axis.plot(
                [row["epoch"] for row in history],
                [row["train_loss"] for row in history],
                style,
                label=group,
            )
        axis.set_title(model_name.upper())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Training loss")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Training loss: Baseline vs Batch Normalization")
    fig.tight_layout()
    fig.savefig(output_dir / "training_loss_curves.png", dpi=180)
    plt.close(fig)


def plot_test_accuracy(histories, output_dir, threshold):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for axis, model_name in zip(axes.flat, MODELS):
        for group, style in (("Baseline", "-"), ("BN", "--")):
            history = histories[(group, model_name)]
            axis.plot(
                [row["epoch"] for row in history],
                [row["test_accuracy"] for row in history],
                style,
                label=group,
            )
        axis.axhline(threshold, color="gray", linestyle=":", linewidth=1, label="80% threshold")
        axis.set_title(model_name.upper())
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Test accuracy")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle("Test accuracy: Baseline vs Batch Normalization")
    fig.tight_layout()
    fig.savefig(output_dir / "test_accuracy_curves.png", dpi=180)
    plt.close(fig)


def plot_train_vs_test(histories, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(19, 8), sharex=True, sharey=True)
    for row_index, group in enumerate(("Baseline", "BN")):
        for column_index, model_name in enumerate(MODELS):
            axis = axes[row_index, column_index]
            history = histories[(group, model_name)]
            epochs = [row["epoch"] for row in history]
            axis.plot(epochs, [row["train_accuracy"] for row in history], label="Train")
            axis.plot(
                epochs,
                [row["test_accuracy"] for row in history],
                linestyle="--",
                label="Test",
            )
            axis.set_title(f"{model_name.upper()} {group}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel("Accuracy")
            axis.grid(alpha=0.25)
            axis.legend()
    fig.suptitle("Train accuracy vs Test accuracy")
    fig.tight_layout()
    fig.savefig(output_dir / "train_vs_test_accuracy.png", dpi=180)
    plt.close(fig)


def plot_convergence(summaries, output_dir, threshold):
    fig, axis = plt.subplots(figsize=(10, 5))
    x_positions = list(range(len(MODELS)))
    width = 0.36
    for offset, group in ((-width / 2, "Baseline"), (width / 2, "BN")):
        group_rows = {row["model"]: row for row in summaries if row["group"] == group}
        values = []
        for model_name in MODELS:
            value = group_rows[model_name]["epoch_to_80_accuracy"]
            values.append(float(value) if value != "" else 0.0)
        bars = axis.bar([x + offset for x in x_positions], values, width, label=group)
        for bar, value in zip(bars, values):
            label = f"{int(value)}" if value else "NR"
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axis.set_xticks(x_positions, [name.upper() for name in MODELS])
    axis.set_ylabel(f"First epoch reaching {threshold:.0%} test accuracy")
    axis.set_title("Convergence speed (NR = not reached)")
    axis.set_ylim(0, 33)
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "convergence_speed.png", dpi=180)
    plt.close(fig)


def generate_comparison(baseline_dir, bn_dir, output_dir, threshold=0.8):
    baseline_dir = Path(baseline_dir).resolve()
    bn_dir = Path(bn_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_summary_path = baseline_dir / "summary.csv"
    bn_summary_path = bn_dir / "summary.csv"
    if not baseline_summary_path.is_file() or not bn_summary_path.is_file():
        raise FileNotFoundError(
            "缺少 Baseline 或 BN summary.csv；请确认 BN 四模型已经全部训练完成。"
        )
    baseline_summary = {row["model"]: row for row in read_csv(baseline_summary_path)}
    bn_summary = {row["model"]: row for row in read_csv(bn_summary_path)}

    histories = {}
    summaries = []
    for model_name in MODELS:
        baseline_history = normalized_history(
            baseline_dir / model_name / "history.csv", "Baseline"
        )
        bn_history = normalized_history(
            bn_dir / f"{model_name}_bn" / "history.csv", "BN"
        )
        histories[("Baseline", model_name)] = baseline_history
        histories[("BN", model_name)] = bn_history
        summaries.append(
            summarize(
                "Baseline",
                model_name,
                baseline_history,
                int(baseline_summary[model_name]["parameters"]),
                float(baseline_summary[model_name]["training_seconds"]),
                threshold,
            )
        )
        summaries.append(
            summarize(
                "BN",
                model_name,
                bn_history,
                int(bn_summary[model_name]["parameters"]),
                float(bn_summary[model_name]["training_seconds"]),
                threshold,
            )
        )

    atomic_write_csv(summaries, output_dir / "comparison_summary.csv")
    plot_training_loss(histories, output_dir)
    plot_test_accuracy(histories, output_dir, threshold)
    plot_train_vs_test(histories, output_dir)
    plot_convergence(summaries, output_dir, threshold)
    print(f"Baseline vs BN 比较结果：{output_dir}", flush=True)
    return summaries


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="生成 Baseline vs BN 对比指标与图表")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=project_dir / "model_type_experiment" / "results",
    )
    parser.add_argument(
        "--bn-dir",
        type=Path,
        default=project_dir / "model_BN_experiment" / "results" / "bn_group",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir / "model_BN_experiment" / "comparison",
    )
    parser.add_argument("--threshold", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.threshold <= 1.0:
        raise ValueError("threshold 必须在 (0, 1] 范围内。")
    generate_comparison(args.baseline_dir, args.bn_dir, args.output_dir, args.threshold)


if __name__ == "__main__":
    main()
