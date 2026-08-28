from pathlib import Path
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
CLASS_NAMES = [
    "airplane",   
    "automobile",  
    "bird",       
    "cat",       
    "deer",        
    "dog",       
    "frog",       
    "horse",     
    "ship",        
    "truck"    
]
CLASS_NAMES_CHINESE = [
    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车"
]

class LeNet(nn.Module):

    def __init__(self, num_classes=10):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.ReLU(),
            nn.Linear(
                in_features=120,
                out_features=84
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=84,
                out_features=num_classes
            )
        )
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("使用设备：", device)

    model = LeNet(num_classes=10)
    model = model.to(device)

    dummy_input = torch.zeros(
        1,          
        3,          
        32,  
        32,       
        device=device
    )

    with torch.no_grad():
        model(dummy_input)

    model_path = Path("./LeNet_CIFAR10.pth")

    if not model_path.exists():
        print(f"没有找到模型文件：{model_path.resolve()}")
        return

    model_state = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(model_state)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    photo_folder = Path("./my_photo")
    if not photo_folder.exists():
        print(f"没有找到图片文件夹：{photo_folder.resolve()}")
        return

    # 找出 my_photo 文件夹内的所有图片
    image_paths = [
        path
        for path in photo_folder.iterdir()
        if path.is_file()
    ]
    image_paths.sort()

    if len(image_paths) == 0:
        print("my_photo 文件夹中没有找到可识别的图片。")
        return
    print(f"共找到 {len(image_paths)} 张图片。\n")

    with torch.no_grad():
        for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                image_tensor = transform(image)
                image_tensor = image_tensor.unsqueeze(0)
                image_tensor = image_tensor.to(device)
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

                # 打印概率最高的三个结果
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
                print('-'*60)

if __name__ == "__main__":
    main()