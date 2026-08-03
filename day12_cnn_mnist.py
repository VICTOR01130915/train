# Day 12：卷积神经网络（CNN）训练 MNIST
# 准确率从 97% → 99%+

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

print("=" * 60)
print("1. 定义 CNN 网络结构")
print("=" * 60)

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层 1：输入 1 通道（灰度图），输出 32 通道，卷积核 3×3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 卷积层 2：输入 32 通道，输出 64 通道，卷积核 3×3
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 池化层：2×2 最大池化，缩小图片尺寸
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层：64 * 7 * 7 → 128 → 10（分类）
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 第一层：卷积 → ReLU → 池化（28→14）
        x = self.pool(self.relu(self.conv1(x)))
        # 第二层：卷积 → ReLU → 池化（14→7）
        x = self.pool(self.relu(self.conv2(x)))
        # 展平：64 * 7 * 7 → 3136 个值
        x = x.view(x.size(0), -1)
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = MNIST_CNN()
print(model)

# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 加载 MNIST 数据（带数据增强）")
print("=" * 60)

# 使用 Day 11 调好的温和增强
transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.RandomAffine(0, scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集：{len(train_dataset)} 张")
print(f"测试集：{len(test_dataset)} 张")

# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 训练 CNN（3 轮，每轮约 1~2 分钟）")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MNIST_CNN().to(device)
print(f"使用设备：{device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 4):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(output, 1)
        total += target.size(0)
        correct += (pred == target).sum().item()

        if batch_idx % 300 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    acc = 100 * correct / total
    print(f"Epoch {epoch} 训练准确率：{acc:.2f}%")

# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 测试集评估")
print("=" * 60)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        _, pred = torch.max(output, 1)
        total += target.size(0)
        correct += (pred == target).sum().item()

acc = 100 * correct / total
print(f"🎯 CNN 测试准确率：{acc:.2f}%")

# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 保存 CNN 模型")
print("=" * 60)

torch.save(model.state_dict(), "mnist_cnn.pth")
print("✅ 模型已保存为 mnist_cnn.pth")

print("\n" + "=" * 60)
print("🎉 Day 12 完成！你拥有了第一个 CNN 模型！")
print(f"   准确率：{acc:.2f}%（MLP 约 97%，提升明显！）")
print("=" * 60)