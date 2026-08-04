# Day 13：迁移学习（Transfer Learning）
# 用预训练模型 ResNet-18 识别 Fashion-MNIST（服装分类）

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

print("=" * 60)
print("1. 加载预训练模型（ResNet-18）")
print("=" * 60)

# 加载在 ImageNet（1000 类，1400 万张图）上预训练的 ResNet-18
model = models.resnet18(weights='IMAGENET1K_V1')

# 冻结所有层（不更新参数）
for param in model.parameters():
    param.requires_grad = False

# 替换最后一层：原本 1000 类 → Fashion-MNIST 10 类
model.fc = nn.Linear(model.fc.in_features, 10)

# 只让最后一层可训练
for param in model.fc.parameters():
    param.requires_grad = True

print("✅ 预训练模型加载完成")
print(f"   - 来源：ImageNet（1400 万张图片，1000 个类别）")
print(f"   - 输出层：已替换为 10 类（Fashion-MNIST）")
print(f"   - 可训练参数：仅最后一层（约 5000 个）")


print("\n" + "=" * 60)
print("2. 加载 Fashion-MNIST 数据集")
print("=" * 60)

# Fashion-MNIST：10 类服装（T恤、牛仔裤、裙子等）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集：{len(train_dataset)} 张")
print(f"测试集：{len(test_dataset)} 张")


print("\n" + "=" * 60)
print("3. 只训练最后一层（约 2~3 分钟）")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"使用设备：{device}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # 只更新最后一层

for epoch in range(1, 4):
    model.train()
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, pred = torch.max(output, 1)
        total += target.size(0)
        correct += (pred == target).sum().item()

        if batch_idx % 300 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    acc = 100 * correct / total
    print(f"Epoch {epoch} 训练准确率：{acc:.2f}%")


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
print(f"🎯 迁移学习测试准确率：{acc:.2f}%")


print("\n" + "=" * 60)
print("5. 保存模型")
print("=" * 60)

torch.save(model.state_dict(), "fashion_mnist_transfer.pth")
print("✅ 模型已保存为 fashion_mnist_transfer.pth")


print("\n" + "=" * 60)
print("🎉 Day 13 完成！")
print(f"   模型：ResNet-18 迁移学习")
print(f"   数据：Fashion-MNIST（服装分类）")
print(f"   准确率：{acc:.2f}%")
print("   你只训练了最后一层，就拿到了 ~90%+ 的准确率！")
print("=" * 60)