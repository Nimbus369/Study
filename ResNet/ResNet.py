import random
import torch
import torch.nn as nn

# 数据集是文件夹形式的
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.models import resnet34, ResNet34_Weights
from torch.utils.data import DataLoader


DATA_ROOT = "./data/flower_data"
MODEL_PATH = "./best_resnet34_flowers.pth"

# hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.0001
VAL_RATIO = 0.2
SEED = 42

NUM_WORKERS = 0

# 是否使用官方 ImageNet 预训练权重
USE_PRETRAINED_WEIGHTS = True

# 第一阶段冻结 ResNet 的卷积主干，只训练最后的分类层
FREEZE_BACKBONE = True

# ImageNet预训练模型使用的 Normalization 参数
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    # 如果使用 GPU，也固定 CUDA 的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_datasets():
    train_transform = transforms.Compose([
        # 随机裁剪并缩放到 ResNet 常用的 224×224, antialias：抗锯齿化
        transforms.RandomResizedCrop(224, antialias=True),
        # 随机水平翻转；花朵左右翻转通常不会改变类别
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_dataset = ImageFolder(DATA_ROOT + "/train", transform=train_transform)
    val_dataset = ImageFolder(DATA_ROOT + "/val", transform=val_transform)

    return train_dataset, val_dataset, train_dataset.classes

def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device,
):
    # 也与 BN 有关
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    # 一批一批读取图片
    for images, labels in data_loader:
        # GPU
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total += batch_size

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return average_loss, accuracy


def evaluate(model, data_loader, loss_function, device):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    # inference_mode 比 no_grad 更彻底
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

def main():
    # Use seed
    seed_everything(SEED)
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
        # persistent_workers=NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pin_memory,
        # persistent_workers=NUM_WORKERS > 0,
    )

    # 创建官方预训练 ResNet-34
    if USE_PRETRAINED_WEIGHTS:
        # transfer learning
        weights = ResNet34_Weights.DEFAULT
        model = resnet34(weights=weights)
    else:
        # 关闭预训练权重时，训练全部参数
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


    # 冻结预训练主干
    if FREEZE_BACKBONE:
        # 先冻结整个模型
        for parameter in model.parameters():
            parameter.requires_grad = False
        # 再打开最后分类层的梯度
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

        print("冻结卷积主干，只训练最后的分类层")
    else:
        print("当前不冻结主干：整个 ResNet 都会参与微调")

    # 模型移到 GPU
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()

    # 只把 requires_grad=True 的参数交给优化器
    trainable_parameters = filter(
        lambda parameter: parameter.requires_grad,
        model.parameters(),
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # 学习率更新
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_accuracy = 0.0

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

        # 根据 val_loss 调整学习率
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

if __name__ == "__main__":
    main()