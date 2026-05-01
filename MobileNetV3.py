import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ----------------------
# 1. MobileNetV3 核心模块
# ----------------------
# Hard-Swish激活函数（V3专用）
class HardSwish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.

# Hard-Sigmoid激活函数（V3专用）
class HardSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.inplace = inplace
    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

# SE注意力模块
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1, bias=True),
            HardSigmoid()
        )
    def forward(self, x):
        scale = self.avg_pool(x)
        scale = self.fc(scale)
        return x * scale

# V3核心：Bneck倒残差模块
class Bneck(nn.Module):
    def __init__(self, in_c, expand_c, out_c, kernel_size, stride, use_se, act):
        super().__init__()
        self.act = HardSwish() if act == 'HS' else nn.ReLU()
        self.use_res_connect = (stride == 1 and in_c == out_c)
        
        layers = []
        # 1x1升维
        if expand_c != in_c:
            layers.extend([
                nn.Conv2d(in_c, expand_c, 1, bias=False),
                nn.BatchNorm2d(expand_c),
                self.act
            ])
        # 深度卷积
        layers.extend([
            nn.Conv2d(expand_c, expand_c, kernel_size, stride, kernel_size//2, groups=expand_c, bias=False),
            nn.BatchNorm2d(expand_c),
            self.act
        ])
        # SE注意力
        if use_se:
            layers.append(SEBlock(expand_c))
        # 1x1降维
        layers.extend([
            nn.Conv2d(expand_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res_connect:
            out += x
        return out

# ----------------------
# 2. MobileNetV3 模型定义
# ----------------------
class MobileNetV3(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # 适配CIFAR10(32x32)：初始卷积步幅=1，不下采样
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            HardSwish()
        )

        # V3-Small 配置（轻量化适配CIFAR10）
        # [输入, 升维, 输出, 核, 步幅, SE, 激活]
        self.bneck_layers = nn.Sequential(
            Bneck(16, 16, 16, 3, 2, True, 'RE'),
            Bneck(16, 72, 24, 3, 2, False, 'RE'),
            Bneck(24, 88, 24, 3, 1, False, 'RE'),
            Bneck(24, 96, 40, 5, 2, True, 'HS'),
            Bneck(40, 240, 40, 5, 1, True, 'HS'),
            Bneck(40, 240, 40, 5, 1, True, 'HS'),
            Bneck(40, 120, 48, 5, 1, True, 'HS'),
            Bneck(48, 144, 48, 5, 1, True, 'HS'),
            Bneck(48, 288, 96, 5, 2, True, 'HS'),
            Bneck(96, 576, 96, 5, 1, True, 'HS'),
            Bneck(96, 576, 96, 5, 1, True, 'HS'),
        )

        # 尾部卷积
        self.last_conv = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            HardSwish()
        )

        # 分类头
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(576, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.bneck_layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x)
        x = x.view(-1, 576)
        x = self.fc(x)
        return x

# ----------------------
# 2. 数据加载
# ----------------------
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ----------------------
# 3. 训练函数
# ----------------------
def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        total_loss += loss.item() * images.size(0)

    avg_train_loss = total_loss / len(train_loader.dataset)
    avg_train_acc = correct / total
    return avg_train_loss, avg_train_acc

# ----------------------
# 4. 测试函数
# ----------------------
def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# ----------------------
# 5. 主程序
# ----------------------
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 64
    lr = 0.01
    num_epochs = 20

    
    model = MobileNetV3(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train_loader, test_loader = get_data_loaders(batch_size)

    # 指标存储
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 训练
    print(f"Training on {device}...")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        test_acc = test(model, test_loader, device)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    # 保存模型
    torch.save(model.state_dict(), 'mobilenetv3_cifar10.pth')
    print("Model saved as mobilenetv3_cifar10.pth")

    # 可视化
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(10, 7))

    plt.plot(epochs, train_loss_list, 'b-', linewidth=2, label='train loss')
    plt.plot(epochs, train_acc_list, 'm--', linewidth=2, label='train acc')
    plt.plot(epochs, test_acc_list, 'g--', linewidth=2, label='test acc')

    plt.xlabel('epoch', fontsize=18)
    plt.xticks(range(2, 11, 2))
    plt.ylim(0, 2.4)
    plt.grid(True)
    plt.legend(loc='upper right', fontsize=18)
    plt.title('MobileNetV3 Training Metrics', fontsize=16)

    plt.savefig('mobilenetv3_training_curve.png', dpi=300, bbox_inches='tight')
    plt.show()