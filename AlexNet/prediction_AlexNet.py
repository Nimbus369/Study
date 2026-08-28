from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import alexnet
from PIL import Image
CLASS_NAMES = [
    "T-shirt/top",   
    "Trouser",       
    "Pullover",      
    "Dress",        
    "Coat",          
    "Sandal",     
    "Shirt",     
    "Sneaker",   
    "Bag",   
    "Ankle boot"  
]
CLASS_NAMES_CHINESE = [
    "T恤或上衣",
    "裤子",
    "套头衫",
    "连衣裙",
    "外套",
    "凉鞋",
    "衬衫",
    "运动鞋",
    "包",
    "短靴"
]
imagenet_mean = (
        0.485,
        0.456,
        0.406
    )

imagenet_std = (
        0.229,
        0.224,
        0.225
    )


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device we use:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = alexnet(weights=None)
    input_features = model.classifier[6].in_features

    model.classifier[6] = nn.Linear(
        in_features=input_features,
        out_features=10
    )

    model_path = Path("./best_alexnet_fashionmnist.pth")

    if not model_path.exists():
        print("没有找到模型文件：")
        print(model_path)
        return

    model_state = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(model_state)
    model = model.to(device)

    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(
            num_output_channels=3
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        )
    ])

    photo_folder = Path("./my_photo")

    if not photo_folder.exists():
        print("没有找到图片文件夹：")
        print(photo_folder)
        return
    image_paths = [
        path
        for path in photo_folder.rglob("*")
        if path.is_file()
    ]
    image_paths.sort()

    if len(image_paths) == 0:
        print("文件夹中没有找到支持的图片。")
        return

    print(f"找到 {len(image_paths)} 张图片。")
    print("-" * 60)

    with torch.inference_mode():
        for image_path in image_paths:
            with Image.open(image_path) as image:
                # 先统一转换成 RGB
                image = image.convert("RGB")
                image_tensor = image_transform(image)

            image_tensor = image_tensor.unsqueeze(0).to(device)

            outputs = model(image_tensor)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            top_probabilities, top_indices = torch.topk(
                probabilities,
                k=3,
                dim=1
            )

            top_probabilities = top_probabilities[0]
            top_indices = top_indices[0]

            print(f"图片：{image_path.name}")

            for rank in range(3):
                class_index = top_indices[rank].item()

                probability = (
                    top_probabilities[rank].item() * 100
                )

                english_name = CLASS_NAMES[class_index]
                chinese_name = CLASS_NAMES_CHINESE[class_index]

                print(
                    f"  第 {rank + 1} 名："
                    f"{chinese_name}（{english_name}），"
                    f"概率：{probability:.2f}%"
                )

            print("-" * 60)

if __name__ == "__main__":
    main()