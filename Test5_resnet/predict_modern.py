"""Predict one image or every image in a directory with a modern checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model_modern import resnet34

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path)
    group.add_argument("--image-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path,
                        default=Path(__file__).resolve().parent / "runs" / "best.pt")
    parser.add_argument("--class-names", type=Path, default=None,
                        help="Optional JSON list; checkpoint class_names is preferred.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def image_paths(args: argparse.Namespace) -> list[Path]:
    if args.image is not None:
        return [args.image]
    assert args.image_dir is not None
    return sorted(
        path for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def load_rgb_tensor(path: Path, preprocess: transforms.Compose) -> torch.Tensor:
    with Image.open(path) as image:
        return preprocess(image.convert("RGB"))


def main() -> None:
    args = parse_args()
    paths = image_paths(args)
    if not paths:
        raise FileNotFoundError("没有找到可预测的图片")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = checkpoint.get("model_state", checkpoint)

    class_names = checkpoint.get("class_names")
    if class_names is None and args.class_names is not None:
        class_names = json.loads(args.class_names.read_text(encoding="utf-8"))
    if class_names is None:
        class_names = [str(i) for i in range(state["fc.weight"].shape[0])]

    model = resnet34(num_classes=len(class_names)).to(device)
    model.load_state_dict(state)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    with torch.inference_mode():
        for start in range(0, len(paths), args.batch_size):
            batch_paths = paths[start:start + args.batch_size]
            batch = torch.stack([
                load_rgb_tensor(path, preprocess) for path in batch_paths
            ]).to(device)
            probabilities = model(batch).softmax(dim=1)
            top_k = min(args.top_k, probabilities.shape[1])
            scores, indices = probabilities.topk(top_k, dim=1)
            for path, row_scores, row_indices in zip(batch_paths, scores, indices):
                predictions = ", ".join(
                    f"{class_names[index.item()]}={score.item():.3f}"
                    for score, index in zip(row_scores, row_indices)
                )
                print(f"{path}: {predictions}")


if __name__ == "__main__":
    main()
