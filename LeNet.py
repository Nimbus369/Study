import torch, torchvision
from torch.utils.data import DataLoader
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()                  #继承父亲class的性质

        self.feature = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=6,
                out_channels=16,
                kernel_size=5
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier=nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(120),              #Lazy：只需要制定输出的向量是多少维即可，输入多少系统自己算
            nn.ReLU(),
            nn.Linear(in_features=120, out_features=84),
            nn.ReLU(),
            nn.Linear(in_features=84, out_features=num_classes),
        )
    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x)
        return x
        
def train_one_epoch(model, data_loader, loss_function, optimizer, device):
    model.train()

    total_loss, total, correct = 0.0,0.0,0.0
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total +=labels.size(0)
        predicted = outputs.argmax(dim=1)        
        correct += (predicted == labels).sum().item()

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy

def evaluate(model, data_loader, loss_function, device):
    model.eval()

    total_loss, total, correct = 0.0,0.0,0.0
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            # optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            # loss.backward()
            # optimizer.step()

            total_loss += loss.item() * images.size(0)
            total +=labels.size(0)
            predicted = outputs.argmax(dim=1)                #
            correct += (predicted == labels).sum().item()

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device we use: ",device)
    transform = torchvision.transforms.Compose([       #Compose is for images preprocessing
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(              # Here, Normalize is for (R,G,B) three channels. In the previous step ToTensor, 
                                                       # [0,255] has been transformed to [0,1]. So Normalize is to [-1, 1]
            mean = (0.5, 0.5, 0.5),
            std = (0.5, 0.5, 0.5)  
        )
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", 
        train = True,
        download = True,
        transform = transform
        )
    test_dataset = torchvision.datasets.CIFAR10(
            root="./data", 
            train = False,
            download = True,
            transform = transform
        )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        batch_size=128,                      #炼丹，一般选2的次方能最大化硬件效率。能选多大取决于硬件水平。
                                            #经验法则：如果batch_size翻倍，对于learning_rate最好也要翻倍。
        num_workers=0                       #还是取决于硬件
    )
    test_loader = DataLoader(
            dataset=test_dataset,
            shuffle=False,
            batch_size=128,                      #炼丹，一般选2的次方能最大化硬件效率。能选多大取决于硬件水平。
                                                #经验法则：如果batch_size翻倍，对于learning_rate最好也要翻倍。
            num_workers=0,                       #还是取决于硬件
        )

    model = LeNet()
    model = model.to(device)

    dummy_input = torch.zeros(          #LazyLinear要看过一遍才能自动计算维度
        1,
        3,
        32,
        32,
        device=device
    )
    with torch.no_grad():       #初始化LazyLinear
        model(dummy_input)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 30

    for epoch in range(epochs):
        train_loss, train_accuracy = train_one_epoch(
            model = model,
            data_loader = train_loader,
            loss_function = loss_function,
            optimizer = optimizer,
            device = device
        )
        test_loss, test_accuracy = evaluate(
            model = model,
            data_loader = test_loader,
            loss_function = loss_function,
            # optimizer = optimizer,
            device = device
        )
        if epoch % 5 == 0:
            print(
                f"第 {epoch + 1} 轮 / 共 {epochs} 轮，"
                f"训练损失：{train_loss:.4f}，"
                f"训练准确率：{train_accuracy:.2f} ，"
                f"测试损失：{test_loss:.4f}，"
                f"测试准确率：{test_accuracy:.2f}"
            )
    torch.save(
        model.state_dict(),
        "LeNet_CIFAR10.pth"
    )
    print("模型已保存为 LeNet_CIFAR10.pth")

if __name__ == "__main__":
    main()
    

