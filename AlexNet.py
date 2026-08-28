# 导入 PyTorch
import torch

# 导入神经网络模块
import torch.nn as nn

# 导入 torchvision，里面包含数据集和经典模型
import torchvision

# 导入图片预处理工具
import torchvision.transforms as transforms

# 导入 DataLoader，用于分批读取数据
from torch.utils.data import DataLoader

# 导入 AlexNet 以及它的预训练权重
from torchvision.models import alexnet, AlexNet_Weights


# FashionMNIST 的十个类别
# CLASS_NAMES = [
#     "T-shirt/top",  # 0：T恤
#     "Trouser",      # 1：裤子
#     "Pullover",     # 2：套头衫
#     "Dress",        # 3：连衣裙
#     "Coat",         # 4：外套
#     "Sandal",       # 5：凉鞋
#     "Shirt",        # 6：衬衫
#     "Sneaker",      # 7：运动鞋
#     "Bag",          # 8：包
#     "Ankle boot"    # 9：短靴
# ]


# 定义训练一个 epoch 的函数
def train_one_epoch(
    model,
    data_loader,
    loss_function,
    optimizer,
    device
):
    # 切换到训练模式
    model.train()

    # 用来记录所有样本的总损失
    total_loss = 0.0

    # 用来记录样本总数
    total = 0

    # 用来记录预测正确的样本数
    correct = 0

    # 一批一批地读取训练数据
    for images, labels in data_loader:

        # 把图片移到GPU
        images = images.to(
            device,
            non_blocking=True
        )
        labels = labels.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad()

        outputs = model(images)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        # 当前批次的图片数量
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total += batch_size

        # 在十个类别中找到分数最高的类别
        predicted = outputs.argmax(dim=1)

        # 累加预测正确的数量
        correct += (predicted == labels).sum().item()

    # 计算整个训练集的平均损失
    average_loss = total_loss / total

    # 计算训练准确率
    accuracy = 100.0 * correct / total

    # 返回训练损失和训练准确率
    return average_loss, accuracy


# 定义模型评估函数
def evaluate(
    model,
    data_loader,
    loss_function,
    device
):
    # 切换到测试模式
    model.eval()
    total_loss = 0.0
    total, correct = 0, 0
    with torch.no_grad():
        for images, labels in data_loader:

            images = images.to(
                device,
                non_blocking=True
            )
            labels = labels.to(
                device,
                non_blocking=True
            )

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

# 定义主函数
def main():

    # 固定随机种子，使每次运行的结果更容易复现
    # torch.manual_seed(42)

    # 如果 CUDA 可用就使用 GPU，否则使用 CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # 打印当前使用的设备
    print("Device we use:", device)

    # 如果使用 GPU，就打印显卡名称
    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

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

    # 定义训练集的图片预处理
    train_transform = transforms.Compose([

        # FashionMNIST 原图是 28×28
        # AlexNet 通常接收 224×224，所以将其放大
        transforms.Resize((224, 224)),

        # FashionMNIST 是单通道灰度图
        # AlexNet 需要三通道，因此复制成三个通道
        transforms.Grayscale(
            num_output_channels=3
        ),

        # 随机水平翻转，用于简单的数据增强
        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        # 把 PIL 图片转换成 PyTorch 张量
        transforms.ToTensor(),

        # 使用 ImageNet 的参数进行标准化
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        )
    ])

    # 定义测试集预处理
    test_transform = transforms.Compose([

        # 测试图片同样要转换成 224×224
        transforms.Resize((224, 224)),

        # 单通道复制成三通道
        transforms.Grayscale(
            num_output_channels=3
        ),

        # 转换成张量
        transforms.ToTensor(),

        # 必须使用和训练时一致的标准化方式
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        )
    ])

    # 创建 FashionMNIST 训练数据集
    train_dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    # 创建 FashionMNIST 测试数据集
    test_dataset = torchvision.datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=test_transform
    )

    # 如果使用 GPU，就开启锁页内存
    use_pin_memory = device.type == "cuda"

    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        # 使用 GPU 时开启锁页内存
        pin_memory=use_pin_memory
    )

    # 获取 AlexNet 的 ImageNet 预训练权重
    weights = AlexNet_Weights.DEFAULT

    # 创建预训练 AlexNet
    # 第一次运行时会自动下载预训练参数
    model = alexnet(weights=weights)

    # 查看 AlexNet 最后一层原来的输入维度
    last_layer_input_features = (
        model.classifier[6].in_features
    )

    # 原始 AlexNet 最后一层输出 1000 类
    # FashionMNIST 只有 10 类，所以替换最后一层
    model.classifier[6] = nn.Linear(
        in_features=last_layer_input_features,
        out_features=10
    )

    # 第一阶段先冻结卷积特征提取部分
    # 冻结后，这些参数不会被训练
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # 把模型移动到 GPU 或 CPU
    model = model.to(device)

    # 打印模型最后一层
    print("新的分类层：")
    print(model.classifier[6])

    # 定义多分类交叉熵损失函数
    loss_function = nn.CrossEntropyLoss()

    # 只把 requires_grad=True 的参数交给优化器
    # 目前主要训练 AlexNet 的分类器部分
    trainable_parameters = filter(
        lambda parameter: parameter.requires_grad,
        model.parameters()
    )

    # 使用 AdamW 优化器
    optimizer = torch.optim.AdamW(
        trainable_parameters,

        # 只训练分类器时可以使用稍大的学习率
        lr=0.0005,

        # 权重衰减可以减轻过拟合
        weight_decay=0.0001
    )

    # 当测试损失连续若干轮没有降低时，减小学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # 最大训练轮数
    epochs = 20

    # 保存当前最佳测试准确率
    best_accuracy = 0.0
    
    # 开始训练
    for epoch in range(epochs):

        # 训练一个 epoch
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            data_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device
        )

        # 在测试集上评估
        test_loss, test_accuracy = evaluate(
            model=model,
            data_loader=test_loader,
            loss_function=loss_function,
            device=device
        )

        # 根据测试损失调整学习率
        scheduler.step(test_loss)

        # 获取当前学习率
        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        # 打印当前训练结果
        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"lr={current_learning_rate:.6f} | "
            f"train loss={train_loss:.4f} | "
            f"train accuracy={train_accuracy:.2f}% | "
            f"test loss={test_loss:.4f} | "
            f"test accuracy={test_accuracy:.2f}%"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy

            torch.save(
                model.state_dict(),
                "best_alexnet_fashionmnist.pth"
            )

            print(
                f"保存最佳模型，准确率："
                f"{best_accuracy:.2f}%"
            )

    print(
        f"\n训练结束，最佳测试准确率："
        f"{best_accuracy:.2f}%"
    )
    print(
        "模型已保存为："
        "best_alexnet_fashionmnist.pth"
    )

if __name__ == "__main__":
    main()