# Path 用于处理文件和文件夹路径
from pathlib import Path

# 导入 PyTorch
import torch

# 导入神经网络模块
import torch.nn as nn

# 导入图片预处理工具
from torchvision import transforms

# 导入 AlexNet 模型
from torchvision.models import alexnet

# PIL 用于读取 JPG、WEBP 等图片
from PIL import Image


# FashionMNIST 的十个类别
# 顺序必须与训练数据集的类别顺序一致
CLASS_NAMES = [
    "T-shirt/top",   # 0：T恤或上衣
    "Trouser",       # 1：裤子
    "Pullover",      # 2：套头衫
    "Dress",         # 3：连衣裙
    "Coat",          # 4：外套
    "Sandal",        # 5：凉鞋
    "Shirt",         # 6：衬衫
    "Sneaker",       # 7：运动鞋
    "Bag",           # 8：包
    "Ankle boot"     # 9：短靴
]


# 对应的中文名称
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


def main():

    # 如果 CUDA 可以使用，就选择 GPU
    # 否则使用 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 打印当前使用的设备
    print("Device we use:", device)

    # 如果正在使用 GPU，就打印显卡名称
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 创建 AlexNet 模型
    # weights=None 表示不重新下载预训练权重
    # 因为之后会加载我们自己训练好的参数
    model = alexnet(weights=None)

    # 取得 AlexNet 最后一层原来的输入维度
    # AlexNet 这里通常是 4096
    input_features = model.classifier[6].in_features

    # 把最后一层替换成 FashionMNIST 的 10 分类
    # 模型结构必须与训练时保持一致
    model.classifier[6] = nn.Linear(
        in_features=input_features,
        out_features=10
    )

    # 训练好的模型文件路径
    model_path = Path(
        r"F:\Study\Coding\deep learning\Project\Lesson1"
        r"\best_alexnet_fashionmnist.pth"
    )

    # 检查模型文件是否存在
    if not model_path.exists():
        print("没有找到模型文件：")
        print(model_path)
        return

    # 读取训练好的模型参数
    model_state = torch.load(
        model_path,
        map_location=device,
        weights_only=True
    )

    # 把训练好的参数载入模型
    model.load_state_dict(model_state)

    # 把模型移动到 GPU 或 CPU
    model = model.to(device)

    # 切换到预测模式
    # 这会关闭 Dropout 的随机效果
    model.eval()

    # ImageNet 预训练模型使用的均值
    imagenet_mean = (
        0.485,
        0.456,
        0.406
    )

    # ImageNet 预训练模型使用的标准差
    imagenet_std = (
        0.229,
        0.224,
        0.225
    )

    # 定义预测图片的预处理
    # 必须尽可能与训练时的处理保持一致
    image_transform = transforms.Compose([

        # AlexNet 的输入大小是 224×224
        transforms.Resize((224, 224)),

        # FashionMNIST 是灰度图
        # 训练时曾把灰度图复制成三个通道
        # 所以预测网上图片时也先转成灰度，再复制为三个通道
        transforms.Grayscale(
            num_output_channels=3
        ),

        # 将 PIL 图片转换成 PyTorch 张量
        transforms.ToTensor(),

        # 使用与训练时相同的标准化参数
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        )
    ])

    # 需要预测的图片文件夹
    photo_folder = Path(
        r"F:\Study\Coding\deep learning\Project\Lesson1"
        r"\my_photo\for AlexNet"
    )

    # 检查图片文件夹是否存在
    if not photo_folder.exists():
        print("没有找到图片文件夹：")
        print(photo_folder)
        return

    # 支持的图片格式
    allowed_suffixes = {
        ".jpg",
        ".jpeg",
        ".webp",
        ".png",
        ".bmp"
    }

    # 查找文件夹中的全部图片
    # rglob("*") 也会查找子文件夹中的图片
    image_paths = [
        path
        for path in photo_folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in allowed_suffixes
    ]

    # 按照文件名排序
    image_paths.sort()

    # 检查是否找到图片
    if len(image_paths) == 0:
        print("文件夹中没有找到支持的图片。")
        return

    # 打印找到的图片数量
    print(f"找到 {len(image_paths)} 张图片。")
    print("-" * 60)

    # inference_mode 表示正在进行预测
    # 它会关闭梯度计算，节省显存并提高速度
    with torch.inference_mode():

        # 逐张处理图片
        for image_path in image_paths:

            try:
                # 打开图片
                with Image.open(image_path) as image:

                    # 某些 WEBP 图片可能包含透明通道
                    # 因此先统一转换成 RGB
                    image = image.convert("RGB")

                    # 对图片进行缩放、灰度化和标准化
                    image_tensor = image_transform(image)

                # 目前图片形状是：
                # [3, 224, 224]
                #
                # 模型需要包含 batch 维度：
                # [1, 3, 224, 224]
                #
                # unsqueeze(0) 会在最前面增加一个维度
                image_tensor = image_tensor.unsqueeze(0)

                # 把图片移动到 GPU 或 CPU
                image_tensor = image_tensor.to(device)

                # 将图片输入模型
                # outputs 中是十个类别的原始分数
                outputs = model(image_tensor)

                # 使用 softmax 将原始分数转换成概率
                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                # 找出概率最高的三个类别
                top_probabilities, top_indices = torch.topk(
                    probabilities,
                    k=3,
                    dim=1
                )

                # 去掉最前面的 batch 维度
                top_probabilities = top_probabilities[0]
                top_indices = top_indices[0]

                # 打印当前图片的文件名
                print(f"图片：{image_path.name}")

                # 打印前三名
                for rank in range(3):

                    # 得到类别编号
                    class_index = top_indices[rank].item()

                    # 得到预测概率，并转换成百分比
                    probability = (
                        top_probabilities[rank].item() * 100
                    )

                    # 根据编号取得英文类别名称
                    english_name = CLASS_NAMES[class_index]

                    # 根据编号取得中文类别名称
                    chinese_name = CLASS_NAMES_CHINESE[class_index]

                    # 打印预测结果
                    print(
                        f"  第 {rank + 1} 名："
                        f"{chinese_name}（{english_name}），"
                        f"概率：{probability:.2f}%"
                    )

                # 不同图片之间打印分隔线
                print("-" * 60)

            # 如果某张图片无法读取，就打印错误并继续下一张
            except Exception as error:
                print(f"无法处理图片：{image_path.name}")
                print(f"错误信息：{error}")
                print("-" * 60)


# 只有直接运行这个文件时才执行 main()
if __name__ == "__main__":
    main()