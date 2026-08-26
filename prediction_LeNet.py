# 导入 Path，用来处理文件夹和文件路径
from pathlib import Path

# 导入 PyTorch
import torch

# 导入神经网络模块
import torch.nn as nn

# 导入图像预处理工具
import torchvision.transforms as transforms

# PIL 用来读取图片
from PIL import Image


# CIFAR-10 的类别名称
# 顺序必须与 CIFAR-10 数据集的标签顺序一致
CLASS_NAMES = [
    "airplane",    # 0：飞机
    "automobile",  # 1：汽车
    "bird",        # 2：鸟
    "cat",         # 3：猫
    "deer",        # 4：鹿
    "dog",         # 5：狗
    "frog",        # 6：青蛙
    "horse",       # 7：马
    "ship",        # 8：船
    "truck"        # 9：卡车
]


# 中文类别名称，只用于显示
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


# 模型结构必须与训练时完全一致
class LeNet(nn.Module):

    def __init__(self, num_classes=10):

        # 初始化父类 nn.Module
        super().__init__()

        # 卷积特征提取部分
        self.feature = nn.Sequential(

            # 第一层卷积
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5
            ),

            # 激活函数
            nn.ReLU(),

            # 最大池化
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            # 第二层卷积
            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),

            # 激活函数
            nn.ReLU(),

            # 最大池化
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )

        # 分类部分
        self.classifier = nn.Sequential(

            # 把特征图展开成向量
            nn.Flatten(),

            # 输入维度自动推断
            nn.LazyLinear(120),

            # 激活函数
            nn.ReLU(),

            # 第二个全连接层
            nn.Linear(
                in_features=120,
                out_features=84
            ),

            # 激活函数
            nn.ReLU(),

            # 最后输出 10 个类别的分数
            nn.Linear(
                in_features=84,
                out_features=num_classes
            )
        )

    # 定义数据如何经过模型
    def forward(self, x):

        # 先提取特征
        x = self.feature(x)

        # 再进行分类
        x = self.classifier(x)

        # 返回分类分数
        return x


def main():

    # 如果有 CUDA 就使用 GPU，否则使用 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 打印当前设备
    print("使用设备：", device)

    # 创建模型
    model = LeNet(num_classes=10)

    # 把模型移动到相应设备
    model = model.to(device)

    # 先输入一个假图片，让 LazyLinear 确定输入维度
    dummy_input = torch.zeros(
        1,            # 一张图片
        3,            # RGB 三通道
        32,           # 高度
        32,           # 宽度
        device=device
    )

    # 初始化时不需要计算梯度
    with torch.no_grad():
        model(dummy_input)

    # 指定模型参数文件的位置
    model_path = Path("./LeNet_CIFAR10.pth")

    # 检查模型文件是否存在
    if not model_path.exists():
        print(f"没有找到模型文件：{model_path.resolve()}")
        return

    # 从文件中读取训练好的参数
    model_state = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    # 把参数装入模型
    model.load_state_dict(model_state)

    # 切换到测试模式
    model.eval()

    # 定义图片预处理
    # 必须尽量和训练时的预处理保持一致
    transform = transforms.Compose([

        # 网上图片大小不统一，需要统一缩放成 32×32
        transforms.Resize((32, 32)),

        # 把 PIL 图片转换成 PyTorch 张量
        transforms.ToTensor(),

        # 使用和训练时相同的标准化参数
        transforms.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5)
        )
    ])

    # 指定图片文件夹
    photo_folder = Path("./my_photo")

    # 检查图片文件夹是否存在
    if not photo_folder.exists():
        print(f"没有找到图片文件夹：{photo_folder.resolve()}")
        return

    # 允许读取的图片格式
    allowed_suffixes = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    # 找出 my_photo 文件夹内的所有图片
    image_paths = [
        path
        for path in photo_folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
    ]

    # 按文件名排序
    image_paths.sort()

    # 如果没有找到图片，就提示用户
    if len(image_paths) == 0:
        print("my_photo 文件夹中没有找到可识别的图片。")
        return

    # 打印图片数量
    print(f"共找到 {len(image_paths)} 张图片。\n")

    # 测试时关闭梯度计算
    with torch.no_grad():

        # 逐张读取和预测图片
        for image_path in image_paths:

            try:
                # 打开图片，并且强制转换成 RGB 三通道
                image = Image.open(image_path).convert("RGB")

                # 缩放、转换张量并标准化
                image_tensor = transform(image)

                # 当前形状是 [3, 32, 32]
                # 模型要求形状是 [批次大小, 3, 32, 32]
                # unsqueeze(0) 会在最前面增加一个批次维度
                image_tensor = image_tensor.unsqueeze(0)

                # 把图片张量移动到 GPU 或 CPU
                image_tensor = image_tensor.to(device)

                # 让模型对图片进行预测
                outputs = model(image_tensor)

                # 把模型输出分数转换成概率
                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                # 找出概率最高的 3 个类别
                top_probabilities, top_indices = torch.topk(
                    probabilities,
                    k=3,
                    dim=1
                )

                # 去掉批次维度
                top_probabilities = top_probabilities[0]
                top_indices = top_indices[0]

                # 打印当前文件名
                print(f"图片：{image_path.name}")

                # 打印概率最高的三个结果
                for rank in range(3):

                    # 取出类别编号
                    class_index = top_indices[rank].item()

                    # 取出概率并转换成百分比
                    probability = (
                        top_probabilities[rank].item() * 100
                    )

                    # 根据编号找到类别名称
                    english_name = CLASS_NAMES[class_index]
                    chinese_name = CLASS_NAMES_CHINESE[class_index]

                    # 打印当前类别
                    print(
                        f"  第 {rank + 1} 名："
                        f"{chinese_name}（{english_name}），"
                        f"概率：{probability:.2f}%"
                    )

                # 每张图片的结果之间空一行
                print()

            # 如果某张图片无法打开，不让整个程序停止
            except Exception as error:
                print(f"读取 {image_path.name} 时出错：{error}")
                print()


# 直接运行 predict.py 时执行 main()
if __name__ == "__main__":
    main()