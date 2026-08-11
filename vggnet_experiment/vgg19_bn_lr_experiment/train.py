"""独立训练 VGG19-BN 的多组学习率实验，不修改已有 VGG 实验。"""

import argparse
import csv
import gc
import json
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import vgg19_bn


HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "test_loss",
    "test_accuracy",
)
DEFAULT_LEARNING_RATES = (0.0002, 0.0003, 0.0005, 0.001, 0.00005)


def parse_args():
    experiment_dir = Path(__file__).resolve().parent
    project_dir = experiment_dir.parent
    parser = argparse.ArgumentParser(description="VGG19-BN 学习率消融实验")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data" / "flower_data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_dir / "results",
    )
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=list(DEFAULT_LEARNING_RATES),
        help="BN学习率分组；已有完整/中断结果可配合--resume跳过或续训",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 0, 4))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="从每组学习率的最后检查点继续")
    parser.add_argument("--no-amp", action="store_true", help="关闭 CUDA 混合精度")
    parser.add_argument("--skip-report", action="store_true", help="训练后不自动生成汇总和报告")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def learning_rate_slug(learning_rate):
    return f"lr_{learning_rate:.10g}".replace(".", "p")


def build_datasets(data_dir):
    train_dir = data_dir / "train"
    test_dir = data_dir / "val"
    if not train_dir.is_dir() or not test_dir.is_dir():
        raise FileNotFoundError(
            f"数据目录不完整：{data_dir}\n需要 train/<类别> 和 val/<类别>。"
        )

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("训练集与留出集的类别不一致。")
    return train_dataset, test_dataset


def build_loaders(train_dataset, test_dataset, args, device):
    # 与已有实验一致，共用同一个固定种子的 generator。
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker,
        "persistent_workers": args.workers > 0,
        "generator": generator,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, test_loader


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


def atomic_torch_save(value, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary_path)
    os.replace(temporary_path, path)


def save_history(rows, path):
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def load_history(path):
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    return [
        {
            "epoch": int(row["epoch"]),
            **{field: float(row[field]) for field in HISTORY_FIELDS if field != "epoch"},
        }
        for row in rows
    ]


def save_config(path, args, learning_rate, train_dataset, test_dataset, device):
    config = {
        "model": "VGG19",
        "batch_norm": True,
        "block_order": "Conv2d -> BatchNorm2d -> ReLU",
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "amp": device.type == "cuda" and not args.no_amp,
        "data_dir": str(args.data_dir),
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "test_source_directory": "val",
        "classes": train_dataset.classes,
        "torch_version": torch.__version__,
        "device": str(device),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)
    os.replace(temporary_path, path)


def validate_resume_config(path, args, learning_rate):
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    expected = {
        "learning_rate": learning_rate,
        "optimizer": "Adam",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "batch_norm": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"不能续训 {path.parent.name}：配置 {key}={config.get(key)!r}，当前为 {value!r}。"
            )


def train_one_learning_rate(learning_rate, train_dataset, test_dataset, args, device):
    set_seed(args.seed)
    run_dir = args.output_dir / learning_rate_slug(learning_rate)
    run_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "history.csv"
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "last_checkpoint.pth"
    best_weights_path = run_dir / "best.pth"

    train_loader, test_loader = build_loaders(train_dataset, test_dataset, args, device)
    model = vgg19_bn(num_classes=len(train_dataset.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    amp_enabled = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    history = []
    best_test_accuracy = -math.inf
    best_epoch = 0
    total_seconds = 0.0

    if args.resume and checkpoint_path.is_file():
        if not config_path.is_file():
            raise FileNotFoundError(f"检查点存在但缺少配置：{config_path}")
        validate_resume_config(config_path, args, learning_rate)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        best_test_accuracy = float(checkpoint["best_test_accuracy"])
        best_epoch = int(checkpoint["best_epoch"])
        total_seconds = float(checkpoint.get("total_seconds", 0.0))
        history = load_history(history_path)
        if len(history) != start_epoch:
            raise RuntimeError(f"{run_dir.name} 的 history.csv 与检查点 epoch 不一致。")
        print(f"[{run_dir.name}] 从 epoch {start_epoch} 继续。", flush=True)
    elif history_path.exists() or checkpoint_path.exists():
        raise FileExistsError(
            f"{run_dir} 已有结果；为防止覆盖，请使用 --resume 或更换 --output-dir。"
        )
    else:
        save_config(config_path, args, learning_rate, train_dataset, test_dataset, device)

    print(
        f"\n[VGG19-BN lr={learning_rate:g}] epochs={args.epochs}, "
        f"batch_size={args.batch_size}, seed={args.seed}, amp={amp_enabled}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, amp_enabled
        )
        test_loss, test_accuracy = run_epoch(
            model, test_loader, criterion, device, amp_enabled=amp_enabled
        )
        epoch_seconds = time.perf_counter() - epoch_start
        total_seconds += epoch_seconds

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        }
        history.append(row)
        save_history(history, history_path)

        if math.isfinite(test_accuracy) and test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            atomic_torch_save(model.state_dict(), best_weights_path)

        checkpoint = {
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_test_accuracy": best_test_accuracy,
            "best_epoch": best_epoch,
            "total_seconds": total_seconds,
            "learning_rate": learning_rate,
            "batch_norm": True,
        }
        atomic_torch_save(checkpoint, checkpoint_path)
        print(
            f"[lr={learning_rate:g}] epoch {epoch + 1:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.4f} | "
            f"test loss {test_loss:.4f}, acc {test_accuracy:.4f} | {epoch_seconds:.1f}s",
            flush=True,
        )

    return run_dir


def main():
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs、batch-size 必须大于0，workers不能小于0。")
    if any(rate <= 0 or not math.isfinite(rate) for rate in args.learning_rates):
        raise ValueError("所有学习率必须为有限正数。")
    if len(set(args.learning_rates)) != len(args.learning_rates):
        raise ValueError("learning-rates 中存在重复值。")

    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_dataset, test_dataset = build_datasets(args.data_dir)
    print(
        f"device={device}, train={len(train_dataset)}, held-out test={len(test_dataset)}",
        flush=True,
    )

    for learning_rate in args.learning_rates:
        train_one_learning_rate(
            learning_rate, train_dataset, test_dataset, args, device
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_report:
        from analyze_results import generate_outputs

        generate_outputs(
            project_dir=Path(__file__).resolve().parent.parent,
            experiment_dir=Path(__file__).resolve().parent,
            required_learning_rates=args.learning_rates,
        )

    print(f"\n学习率实验完成：{args.output_dir}")


if __name__ == "__main__":
    main()
