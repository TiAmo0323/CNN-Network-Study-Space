"""将 flower_photos 按固定种子划分为训练集和验证集。"""

import argparse
import random
import shutil
from pathlib import Path


def parse_args():
    project_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="划分五分类花卉数据集")
    parser.add_argument(
        "--source",
        type=Path,
        default=project_dir / "data" / "flower_data" / "flower_photos",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_dir / "data" / "flower_data",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"未找到解压后的数据集：{source}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val-ratio 必须在 0 和 1 之间。")

    train_root = output / "train"
    val_root = output / "val"
    if train_root.exists() or val_root.exists():
        raise FileExistsError(
            f"{train_root} 或 {val_root} 已存在；为避免误删数据，脚本不会覆盖。"
        )

    classes = sorted(path for path in source.iterdir() if path.is_dir())
    if not classes:
        raise RuntimeError(f"{source} 中没有类别目录。")

    rng = random.Random(args.seed)
    train_count = 0
    val_count = 0
    for class_dir in classes:
        images = sorted(path for path in class_dir.iterdir() if path.is_file())
        val_names = set(rng.sample(images, k=int(len(images) * args.val_ratio)))
        (train_root / class_dir.name).mkdir(parents=True, exist_ok=False)
        (val_root / class_dir.name).mkdir(parents=True, exist_ok=False)
        for image_path in images:
            if image_path in val_names:
                destination = val_root / class_dir.name / image_path.name
                val_count += 1
            else:
                destination = train_root / class_dir.name / image_path.name
                train_count += 1
            shutil.copy2(image_path, destination)
        print(
            f"{class_dir.name}: total={len(images)}, "
            f"train={len(images) - len(val_names)}, val={len(val_names)}"
        )

    print(f"划分完成：train={train_count}, val={val_count}, output={output}")


if __name__ == "__main__":
    main()
