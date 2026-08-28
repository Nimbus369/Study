"""
使用 ResNet-34 进行花卉分类
============================

这份代码演示的是迁移学习（transfer learning）：

1. 从 torchvision 读取官方 ImageNet 预训练的 ResNet-34；
2. 保留它已经学会的通用视觉特征；
3. 把原来输出 1000 个 ImageNet 类别的最后一层，换成花卉分类层；
4. 第一阶段冻结卷积主干，只训练新的分类层。

数据目录默认是：

    data/flower_photos/
    ├── daisy/
    ├── dandelion/
    ├── roses/
    ├── sunflowers/
    └── tulips/

项目中已经有 data/flower_photos.tgz，第一次运行前请先解压：

    tar -xzf data/flower_photos.tgz -C data

这份脚本的写法特意保持和 LeNet.py、AlexNet.py 类似：
训练函数 -> 评估函数 -> main() -> 保存最佳模型。
"""

# Path 比手写字符串路径更稳妥：Windows、Linux 都可以正常工作
from pathlib import Path
import random

# PyTorch 主包
import torch

# 神经网络模块
import torch.nn as nn

# ImageFolder：按照“一个文件夹代表一个类别”的方式读取图片
from torchvision.datasets import ImageFolder

# 图片预处理工具
from torchvision import transforms

# 官方 ResNet-34 和对应的 ImageNet 预训练权重
from torchvision.models import resnet34, ResNet34_Weights

# DataLoader：把数据集分成一个个 batch
from torch.utils.data import DataLoader, Subset


# ============================================================
# 一、可以直接修改的配置
# ============================================================

# 当前脚本所在项目目录
PROJECT_ROOT = Path(__file__).resolve().parent

# 花卉数据集目录
DATA_ROOT = PROJECT_ROOT / "data" / "flower_photos"

# 训练好的模型保存位置
MODEL_PATH = PROJECT_ROOT / "best_resnet34_flowers.pth"

# 超参数
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
VAL_RATIO = 0.2
SEED = 42

# Windows 下先使用 0 最省心；Linux 可以根据 CPU 情况改成 2、4 等
NUM_WORKERS = 0

# 是否使用官方 ImageNet 预训练权重
# True：第一次运行时会自动下载权重（之后会使用本地缓存）
USE_PRETRAINED_WEIGHTS = True

# 第一阶段冻结 ResNet 的卷积主干，只训练最后的分类层
FREEZE_BACKBONE = True


# ImageNet 预训练模型使用的标准化参数
# 训练和预测必须使用同一组参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ============================================================
# 二、准备数据
# ============================================================

def seed_everything(seed):
    """固定随机种子，让每次划分验证集的结果尽量一致。"""

    random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用 GPU，也固定 CUDA 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transforms():
    """创建训练集和验证集的图片预处理。"""

    # 训练集可以做随机增强，帮助模型减少过拟合
    train_transform = transforms.Compose([
        # 随机裁剪并缩放到 ResNet 常用的 224×224
        transforms.RandomResizedCrop(224, antialias=True),

        # 随机水平翻转；花朵左右翻转通常不会改变类别
        transforms.RandomHorizontalFlip(p=0.5),

        # PIL 图片 -> [C, H, W] 的 PyTorch Tensor，像素变成 0~1
        transforms.ToTensor(),

        # 使用 ImageNet 的均值和标准差进行标准化
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # 验证集不能使用随机增强，否则每次评估可能得到不同结果
    val_transform = transforms.Compose([
        # 先缩放短边到 256
        transforms.Resize(256, antialias=True),

        # 从中心裁剪出 224×224
        transforms.CenterCrop(224),

        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_transform, val_transform


def make_stratified_split_indices(targets, val_ratio, seed):
    """
    按类别划分训练集和验证集。

    targets 是 ImageFolder 保存的标签列表，例如：
        [0, 0, 0, 1, 1, 2, ...]

    如果直接对全部图片随机切分，某个类别可能在验证集中比例不合理。
    这里每个类别单独抽取一部分图片，简单地保持类别比例。
    """

    if not 0 < val_ratio < 1:
        raise ValueError("VAL_RATIO 必须在 0 和 1 之间")

    # 使用独立的随机对象，不影响其他地方的随机数
    generator = random.Random(seed)

    train_indices = []
    val_indices = []

    # sorted 保证类别处理顺序稳定
    class_ids = sorted(set(targets))
    for class_id in class_ids:
        # 找到当前类别的所有图片下标
        class_indices = [
            index for index, target in enumerate(targets)
            if target == class_id
        ]

        generator.shuffle(class_indices)

        # 每个类别至少留 1 张图片给验证集，但也必须保证训练集还有图片
        if len(class_indices) < 2:
            raise ValueError(
                f"类别编号 {class_id} 的图片少于 2 张，无法自动划分训练集和验证集"
            )

        val_count = round(len(class_indices) * val_ratio)
        val_count = max(1, val_count)
        val_count = min(len(class_indices) - 1, val_count)

        val_indices.extend(class_indices[:val_count])
        train_indices.extend(class_indices[val_count:])

    # 再打乱一次，让训练 batch 不按照类别集中出现
    generator.shuffle(train_indices)
    generator.shuffle(val_indices)

    return train_indices, val_indices


def create_datasets():
    """读取 ImageFolder 数据，并返回训练集、验证集和类别名称。"""
    train_transform, val_transform = make_transforms()

    # 如果目录本身已经是 train/val 两级结构，就直接使用
    train_dir = DATA_ROOT / "train"
    val_dir = DATA_ROOT / "val"
    if train_dir.is_dir() and val_dir.is_dir():
        train_dataset = ImageFolder(train_dir, transform=train_transform)
        val_dataset = ImageFolder(val_dir, transform=val_transform)

        if train_dataset.classes != val_dataset.classes:
            raise ValueError("训练集和验证集的类别文件夹不一致")

        return train_dataset, val_dataset, train_dataset.classes

    # 当前项目的 flower_photos.tgz 是“每个类别一个文件夹”的扁平结构。
    # 这里创建两个 ImageFolder：
    # - 第一个使用训练增强
    # - 第二个使用验证预处理
    # 两者使用完全相同的下标划分，因此验证图片不会被训练增强污染。
    train_full_dataset = ImageFolder(DATA_ROOT, transform=train_transform)
    val_full_dataset = ImageFolder(DATA_ROOT, transform=val_transform)

    train_indices, val_indices = make_stratified_split_indices(
        train_full_dataset.targets,
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    train_dataset = Subset(train_full_dataset, train_indices)
    val_dataset = Subset(val_full_dataset, val_indices)

    return train_dataset, val_dataset, train_full_dataset.classes


# ============================================================
# 三、训练和评估函数
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
):
    """训练一个 epoch，并返回平均损失和准确率。"""

    # train() 会打开 Dropout，并让 BatchNorm 使用训练模式
    model.train()

    # 如果主干被冻结，BatchNorm 的参数虽然不会求梯度，
    # 但 train() 仍可能更新它的 running mean/running variance。
    # 这里把主干切换到 eval()，使“冻结主干”真正保持固定。
    if FREEZE_BACKBONE:
        model.conv1.eval()
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()
        model.layer4.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    # 一批一批读取图片
    for images, labels in data_loader:
        # 把数据移动到 GPU 或 CPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # 清空上一批次的梯度
        # set_to_none=True 通常比填充 0 更节省一点内存和时间
        optimizer.zero_grad(set_to_none=True)

        # 前向传播：得到每个类别的原始分数 logits
        outputs = model(images)

        # CrossEntropyLoss 内部已经包含了 softmax，
        # 训练时不要提前手动 softmax
        loss = loss_function(outputs, labels)

        # 反向传播：计算梯度
        loss.backward()

        # 根据梯度更新参数
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        # logits 最大的下标就是模型预测的类别
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def evaluate(model, data_loader, loss_function, device):
    """在验证集上评估模型，不计算梯度。"""

    # eval() 会关闭 Dropout 的随机效果，
    # 并让 BatchNorm 使用已经统计好的均值和方差
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    # inference_mode 比 no_grad 更彻底地关闭推理过程中的额外记录
    with torch.inference_mode():
        for images, labels in data_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = loss_function(outputs, labels)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


# ============================================================
# 四、主程序
# ============================================================

def main():
    # 固定随机种子，方便复现实验
    seed_everything(SEED)

    # 有 CUDA 就使用 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device we use:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # 创建训练集和验证集
    train_dataset, val_dataset, class_names = create_datasets()
    print("Classes:", class_names)
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")

    # GPU 使用 pin_memory，可以稍微加快 CPU -> GPU 的数据传输
    use_pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        persistent_workers=NUM_WORKERS > 0,
    )

    # --------------------------------------------------------
    # 创建官方预训练 ResNet-34
    # --------------------------------------------------------

    if USE_PRETRAINED_WEIGHTS:
        # DEFAULT 表示 torchvision 当前推荐的官方权重。
        # 第一次运行会下载，之后通常会从本地缓存读取。
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
    else:
        # 关闭预训练权重时，模型会从随机参数开始，
        # 这就不是迁移学习了，主要用于做对照实验。
        model = resnet34(weights=None)

    # 官方 ResNet-34 原本是 ImageNet 1000 分类
    # model.fc 就是最后的全连接分类层
    input_features = model.fc.in_features
    print("原来的分类层：", model.fc)

    # 花卉数据集有几个类别，就把输出改成几个
    model.fc = nn.Linear(
        in_features=input_features,
        out_features=len(class_names),
    )
    print("新的分类层：", model.fc)

    # --------------------------------------------------------
    # 冻结预训练主干
    # --------------------------------------------------------

    if FREEZE_BACKBONE:
        # 先冻结整个模型
        for parameter in model.parameters():
            parameter.requires_grad = False

        # 再打开最后分类层的梯度
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

        print("当前采用迁移学习第一阶段：冻结卷积主干，只训练 fc 分类层")
    else:
        print("当前不冻结主干：整个 ResNet 都会参与微调")

    # 把模型移动到 GPU 或 CPU
    model = model.to(device)

    # 多分类问题使用交叉熵损失
    loss_function = nn.CrossEntropyLoss()

    # 只把 requires_grad=True 的参数交给优化器
    # 冻结主干时，这里实际上只有 model.fc 的参数
    trainable_parameters = filter(
        lambda parameter: parameter.requires_grad,
        model.parameters(),
    )

    # AdamW 是 Adam 加权重衰减的版本，适合迁移学习起步
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 如果验证损失连续几轮没有下降，就降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_accuracy = 0.0

    # --------------------------------------------------------
    # 开始训练
    # --------------------------------------------------------

    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_accuracy = evaluate(
            model=model,
            data_loader=val_loader,
            loss_function=loss_function,
            device=device,
        )

        # 根据验证损失调整学习率
        scheduler.step(val_loss)

        current_learning_rate = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"lr={current_learning_rate:.6f} | "
            f"train loss={train_loss:.4f} | "
            f"train accuracy={train_accuracy:.2f}% | "
            f"val loss={val_loss:.4f} | "
            f"val accuracy={val_accuracy:.2f}%"
        )

        # 只保存验证集上表现最好的模型
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

            # 不仅保存参数，也保存类别名称，预测脚本可以直接读取
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "best_accuracy": best_accuracy,
            }
            torch.save(checkpoint, MODEL_PATH)

            print(f"保存最佳模型，验证准确率：{best_accuracy:.2f}%")

    print("\n训练结束")
    print(f"最佳验证准确率：{best_accuracy:.2f}%")
    print(f"模型已保存为：{MODEL_PATH}")


# 只有直接运行 python ResNet.py 时，才执行 main()
if __name__ == "__main__":
    main()
