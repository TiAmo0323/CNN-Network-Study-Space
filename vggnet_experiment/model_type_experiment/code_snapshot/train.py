"""在同一实验条件下训练 VGG11/13/16/19，并保存消融实验结果。"""

import argparse
import csv
import gc
import json
import os
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import cfgs, vgg


ALL_MODELS = ("vgg11", "vgg13", "vgg16", "vgg19")
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "val_loss",
    "val_accuracy",
    "epoch_seconds",
)


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="VGG 深度消融实验")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir.parent.parent / "data" / "flower_data",
        help="包含 train/ 和 val/ 的数据集目录",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_dir.parent / "results",
        help="实验结果保存目录",
    )
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 0, 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="从各模型的最后检查点续训")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度训练")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 固定随机性，使模型间比较更稳定。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def build_datasets(data_dir):
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            f"数据目录不完整：{data_dir}\n"
            "需要存在 train/<类别>/图片 和 val/<类别>/图片。"
        )

    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    }
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transform["train"])
    val_dataset = datasets.ImageFolder(val_dir, transform=data_transform["val"])
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("训练集和验证集的类别目录不一致。")
    return train_dataset, val_dataset


def build_loaders(train_dataset, val_dataset, args, device):
    # 每个模型使用同一个独立随机种子，保证 shuffle 和数据增强尽量一致。
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, amp_enabled=False):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    grad_context = torch.enable_grad if is_training else torch.no_grad
    with grad_context():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if is_training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if is_training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_history(history, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    os.replace(temporary_path, path)


def load_history(path):
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    history = []
    for row in rows:
        history.append(
            {
                "epoch": int(row["epoch"]),
                **{key: float(row[key]) for key in HISTORY_FIELDS if key != "epoch"},
            }
        )
    return history


def atomic_torch_save(value, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def train_model(model_name, train_dataset, val_dataset, args, device):
    set_seed(args.seed)
    model_dir = args.output_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    history_path = model_dir / "history.csv"
    checkpoint_path = model_dir / "last_checkpoint.pth"
    best_weights_path = model_dir / "best.pth"

    train_loader, val_loader = build_loaders(train_dataset, val_dataset, args, device)
    model = vgg(model_name=model_name, num_classes=len(train_dataset.classes), init_weights=True)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    start_epoch = 0
    best_val_accuracy = -1.0
    best_epoch = 0
    total_seconds = 0.0
    history = []

    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"]
        best_val_accuracy = checkpoint["best_val_accuracy"]
        best_epoch = checkpoint["best_epoch"]
        total_seconds = checkpoint.get("total_seconds", 0.0)
        history = load_history(history_path)
        print(f"[{model_name}] 从 epoch {start_epoch} 继续训练。", flush=True)
    elif history_path.exists():
        raise FileExistsError(
            f"{history_path} 已存在。请更换 --output-dir，或使用 --resume 继续训练。"
        )

    print(
        f"\n[{model_name}] parameters={parameter_count:,}, epochs={args.epochs}, "
        f"batch_size={args.batch_size}, amp={amp_enabled}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_enabled
        )
        val_loss, val_accuracy = run_epoch(
            model, val_loader, criterion, device, amp_enabled=amp_enabled
        )
        epoch_seconds = time.perf_counter() - epoch_start
        total_seconds += epoch_seconds

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_seconds": epoch_seconds,
        }
        history.append(row)
        save_history(history, history_path)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            atomic_torch_save(model.state_dict(), best_weights_path)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val_accuracy": best_val_accuracy,
            "best_epoch": best_epoch,
            "total_seconds": total_seconds,
            "model_name": model_name,
            "classes": train_dataset.classes,
            "args": vars(args),
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        print(
            f"[{model_name}] epoch {epoch + 1:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.4f} | "
            f"{epoch_seconds:.1f}s",
            flush=True,
        )

    if not history:
        history = load_history(history_path)
    final = history[-1]
    return {
        "model": model_name,
        "parameters": parameter_count,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "final_train_loss": final["train_loss"],
        "final_train_accuracy": final["train_accuracy"],
        "final_val_loss": final["val_loss"],
        "final_val_accuracy": final["val_accuracy"],
        "training_seconds": total_seconds,
    }


def plot_histories(output_dir, model_names):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    found = False
    for model_name in model_names:
        history = load_history(output_dir / model_name / "history.csv")
        if not history:
            continue
        found = True
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [row["train_loss"] for row in history], label=f"{model_name} train")
        axes[0].plot(
            epochs, [row["val_loss"] for row in history], linestyle="--", label=f"{model_name} val"
        )
        axes[1].plot(
            epochs, [row["train_accuracy"] for row in history], label=f"{model_name} train"
        )
        axes[1].plot(
            epochs,
            [row["val_accuracy"] for row in history],
            linestyle="--",
            label=f"{model_name} val",
        )
    if not found:
        plt.close(fig)
        return
    axes[0].set(title="VGG ablation: loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axes[1].set(title="VGG ablation: accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_curves.png", dpi=180)
    plt.close(fig)


def save_summary(rows, path):
    if not rows:
        return
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs、batch-size 必须大于 0，workers 不能小于 0。")

    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    train_dataset, val_dataset = build_datasets(args.data_dir)

    class_indices = {str(index): name for index, name in enumerate(train_dataset.classes)}
    with (args.output_dir / "class_indices.json").open("w", encoding="utf-8") as file:
        json.dump(class_indices, file, indent=2, ensure_ascii=False)
    config = {
        **vars(args),
        "data_dir": str(args.data_dir),
        "output_dir": str(args.output_dir),
        "device": str(device),
        "torch_version": torch.__version__,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "classes": train_dataset.classes,
        "vgg_configs": {name: cfgs[name] for name in args.models},
    }
    with (args.output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)

    print(
        f"device={device}, train={len(train_dataset)}, val={len(val_dataset)}, "
        f"classes={train_dataset.classes}",
        flush=True,
    )
    summaries = []
    for model_name in args.models:
        summaries.append(train_model(model_name, train_dataset, val_dataset, args, device))
        save_summary(summaries, args.output_dir / "summary.csv")
        plot_histories(args.output_dir, args.models)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n消融实验完成：")
    for row in summaries:
        print(
            f"{row['model']}: best val acc={row['best_val_accuracy']:.4f} "
            f"(epoch {row['best_epoch']}), time={row['training_seconds'] / 60:.1f} min"
        )
    print(f"结果目录：{args.output_dir}")


if __name__ == "__main__":
    main()
