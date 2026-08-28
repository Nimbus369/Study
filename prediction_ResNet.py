"""
使用训练好的 ResNet-34 对图片进行预测。

训练脚本会生成：

    best_resnet34_flowers.pth

请把要预测的图片放进下面配置的文件夹：

    my_photo/for ResNet/

也可以直接修改 PHOTO_FOLDER 指向其他目录。
"""

from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet34


# 当前项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 训练脚本保存的模型文件
MODEL_PATH = PROJECT_ROOT / "best_resnet34_flowers.pth"

# 要预测的图片文件夹
PHOTO_FOLDER = PROJECT_ROOT / "my_photo" / "for ResNet"

# 支持的图片格式
ALLOWED_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


def main():
    # 有 GPU 就使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device we use:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 检查模型文件
    if not MODEL_PATH.exists():
        print("没有找到训练好的模型文件：")
        print(MODEL_PATH)
        print("请先运行 ResNet.py 完成训练。")
        return

    # 检查图片文件夹
    if not PHOTO_FOLDER.exists():
        print("没有找到图片文件夹：")
        print(PHOTO_FOLDER)
        print("请创建这个文件夹，或者修改 prediction_ResNet.py 中的 PHOTO_FOLDER。")
        return

    # 查找文件夹及其子文件夹中的全部图片
    image_paths = [
        path
        for path in PHOTO_FOLDER.rglob("*")
        if path.is_file() and path.suffix.lower() in ALLOWED_SUFFIXES
    ]
    image_paths.sort()

    if len(image_paths) == 0:
        print("文件夹中没有找到支持的图片。")
        return

    print(f"找到 {len(image_paths)} 张图片。")
    print("-" * 60)

    # 读取 checkpoint
    # weights_only=True 可以更安全地读取纯参数和基础 Python 数据
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    # 训练脚本保存的是字典；下面同时兼容“直接保存 state_dict”的写法
    if "model_state_dict" in checkpoint:
        model_state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint["class_names"]
    else:
        model_state_dict = checkpoint
        # 如果 checkpoint 没有类别名称，就使用数据集的默认顺序
        class_names = [
            "daisy",
            "dandelion",
            "roses",
            "sunflowers",
            "tulips",
        ]

    # 创建一个没有再次下载权重的 ResNet-34
    model = resnet34(weights=None)

    # 预测模型的最后一层必须和训练时完全一致
    input_features = model.fc.in_features
    model.fc = nn.Linear(
        in_features=input_features,
        out_features=len(class_names),
    )

    # 加载训练好的参数
    model.load_state_dict(model_state_dict)
    model = model.to(device)

    # 切换到评估模式
    model.eval()

    # 预测时必须使用和验证集相同的预处理
    image_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    # inference_mode 会关闭梯度计算，节省显存并提高预测速度
    with torch.inference_mode():
        for image_path in image_paths:
            try:
                # 打开图片后统一转换为 RGB，避免灰度图或透明图通道数不一致
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    image_tensor = image_transform(image)

                # 单张图片经过预处理后是 [3, 224, 224]
                # 模型需要 batch 维度，因此变成 [1, 3, 224, 224]
                image_tensor = image_tensor.unsqueeze(0).to(device)

                # 输出形状为 [1, 类别数]
                outputs = model(image_tensor)

                # dim=1 表示在类别这一维上计算概率
                probabilities = torch.softmax(outputs, dim=1)

                # 取得概率最高的前 3 个类别
                top_k = min(3, len(class_names))
                top_probabilities, top_indices = torch.topk(
                    probabilities,
                    k=top_k,
                    dim=1,
                )

                print(f"图片：{image_path.name}")

                for rank in range(top_k):
                    # [0, rank] 取出这一张图片的第 rank 名结果
                    class_index = top_indices[0, rank].item()
                    probability = top_probabilities[0, rank].item() * 100
                    class_name = class_names[class_index]

                    print(
                        f"  Top {rank + 1}: "
                        f"{class_name:<12} "
                        f"{probability:.2f}%"
                    )

                print("-" * 60)

            except Exception as error:
                # 单张图片损坏时，跳过它并继续预测其他图片
                print(f"无法处理图片 {image_path.name}：{error}")


if __name__ == "__main__":
    main()
