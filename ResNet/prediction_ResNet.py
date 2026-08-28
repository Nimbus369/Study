import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet34
from pathlib import Path

MODEL_PATH = Path("./best_resnet34_flowers.pth")
PHOTO_FOLDER = Path("./my_photo/for ResNet")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device we use:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # check
    if not MODEL_PATH.exists():
        print("没有找到训练好的模型文件：")
        return
    if not PHOTO_FOLDER.exists():
        print("没有找到图片文件夹：")
        return

    # 读图
    image_paths = [
        path
        for path in PHOTO_FOLDER.rglob("*")
        if path.is_file()
    ]
    image_paths.sort()

    if len(image_paths) == 0:
        print("文件夹中没有找到支持的图片。")
        return
    else:
        print(f"找到 {len(image_paths)} 张图片。")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model_state_dict = checkpoint["model_state_dict"]
    class_names = checkpoint["class_names"]

    # 创建一个没有再次下载权重的 ResNet-34
    model = resnet34(weights=None)
    # 修改结构
    input_features = model.fc.in_features
    model.fc = nn.Linear(
        in_features=input_features,
        out_features=len(class_names),
    )

    model.load_state_dict(model_state_dict)
    model = model.to(device)


    model.eval()
    image_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    with torch.inference_mode():
        for image_path in image_paths:
            # 统一转换为 RGB
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = image_transform(image)

            # .unsqueeze 为图片添加 batch 维度
            image_tensor = image_tensor.unsqueeze(0).to(device)

            outputs = model(image_tensor)
            
            # 取得概率最高的前 3 个类别
            probabilities = torch.softmax(outputs, dim=1)
            top_k = 3
            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=top_k,
                dim=1,
            )

            print(f"图片：{image_path.name}")

            for rank in range(top_k):
                class_index = top_indices[0, rank].item()
                probability = top_probabilities[0, rank].item() * 100
                class_name = class_names[class_index]

                print(
                    f"  Top {rank + 1}: "
                    f"{class_name:<12} "
                    f"{probability:.2f}%"
                )

            print("-" * 60)

if __name__ == "__main__":
    main()