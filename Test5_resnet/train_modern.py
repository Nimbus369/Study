"""Train the readable ResNet-34 on a flower ImageFolder dataset.

It accepts either:
  data_root/train/<class>/*.jpg and data_root/val/<class>/*.jpg
or a flat ImageFolder root:
  data_root/<class>/*.jpg

For the flat form, a deterministic, class-balanced validation split is made
without copying image files.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import ImageFile
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.models import ResNet34_Weights
try:
    from tqdm import tqdm
except ImportError:  # tqdm is optional; the training loop remains usable without it.
    def tqdm(iterable, **_kwargs):
        return iterable

from model_modern import resnet34

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "flower_photos")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="0 is the most reliable setting on Windows.")
    parser.add_argument("--seed", type=int, default=42)
    weights = parser.add_mutually_exclusive_group()
    weights.add_argument("--imagenet-weights", action="store_true",
                         help="Download/cache the current official ImageNet weights.")
    weights.add_argument("--pretrained-path", type=Path, default=None,
                         help="Use a local ResNet-34 ImageNet state_dict.")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform


def split_indices(targets: Sequence[int], val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not 0 < val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    rng = random.Random(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_id in sorted(set(targets)):
        class_indices = [index for index, target in enumerate(targets) if target == class_id]
        rng.shuffle(class_indices)
        if len(class_indices) < 2:
            raise ValueError(f"类别 {class_id} 至少需要 2 张图片才能自动划分训练/验证集")
        val_count = min(len(class_indices) - 1, max(1, round(len(class_indices) * val_ratio)))
        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def build_datasets(data_dir: Path, val_ratio: float, seed: int):
    if not data_dir.exists():
        archive = data_dir.with_suffix(".tgz")
        raise FileNotFoundError(
            f"数据目录不存在: {data_dir}\n"
            f"请先解压 {archive}，例如：tar -xzf {archive} -C {data_dir.parent}"
        )

    train_transform, val_transform = make_transforms()
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        return (
            datasets.ImageFolder(train_dir, transform=train_transform),
            datasets.ImageFolder(val_dir, transform=val_transform),
        )

    # ImageFolder is instantiated twice so train and validation can have
    # different transforms while sharing exactly the same split indices.
    train_full = datasets.ImageFolder(data_dir, transform=train_transform)
    val_full = datasets.ImageFolder(data_dir, transform=val_transform)
    train_indices, val_indices = split_indices(train_full.targets, val_ratio, seed)
    return Subset(train_full, train_indices), Subset(val_full, val_indices)


def load_model(
    num_classes: int,
    imagenet_weights: bool,
    pretrained_path: Path | None,
) -> nn.Module:
    if not imagenet_weights and pretrained_path is None:
        return resnet34(num_classes=num_classes)

    # Load the 1000-class architecture first, then replace the classifier.
    model = resnet34(num_classes=1000)
    if imagenet_weights:
        state = ResNet34_Weights.DEFAULT.get_state_dict(progress=True, check_hash=True)
    else:
        assert pretrained_path is not None
        state = torch.load(pretrained_path, map_location="cpu", weights_only=True)
        if "model_state" in state:
            state = state["model_state"]
    model.load_state_dict(state)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def run_epoch(model, loader, loss_fn, optimizer, device, training: bool) -> tuple[float, float]:
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        progress = tqdm(loader, leave=False, desc="train" if training else "valid")
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch_size
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(loss=f"{loss.item():.3f}")

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_dataset, val_dataset = build_datasets(args.data_dir, args.val_ratio, args.seed)
    class_names = train_dataset.dataset.classes if isinstance(train_dataset, Subset) else train_dataset.classes
    val_class_names = val_dataset.dataset.classes if isinstance(val_dataset, Subset) else val_dataset.classes
    if class_names != val_class_names:
        raise ValueError("训练集和验证集的类别目录不一致")

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    print(f"classes: {class_names}")
    print(f"images: train={len(train_dataset)}, val={len(val_dataset)}")

    model = load_model(
        len(class_names), args.imagenet_weights, args.pretrained_path
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    best_accuracy = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, optimizer, device, True)
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, optimizer, device, False)
        scheduler.step()
        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_accuracy:.3f} | "
            f"val loss {val_loss:.4f}, acc {val_accuracy:.3f}"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_accuracy": val_accuracy,
            "class_names": class_names,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(checkpoint, args.output_dir / "best.pt")

    print(f"best validation accuracy: {best_accuracy:.3f}")
    print(f"checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
