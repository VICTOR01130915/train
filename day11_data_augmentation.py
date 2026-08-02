# Day 11：数据增强（Data Augmentation）
# 让模型见过更多“变形”的图片 → 泛化能力更强

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

print("=" * 60)
print("1. 定义网络结构（同前两天的 MLP）")
print("=" * 60)

class MNIST_MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# ------------------------------------------------------------
print("\n" + "=" * 60)
print("2. 定义两种数据预处理方式")
print("=" * 60)

# 方式 A：无数据增强（只有归一化）
transform_no_aug = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 方式 B：带数据增强（随机旋转 + 平移 + 缩放）
transform_with_aug = transforms.Compose([
    transforms.RandomRotation(10),                # 旋转 ±10°（原来 15°）
    transforms.RandomAffine(0, translate=(0.05, 0.05)),  # 平移 5%（原来 15%）
    transforms.RandomAffine(0, scale=(0.9, 1.1)),        # 缩放 90%~110%（原来 85%~115%）
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载数据集（只下载一次，复用）
train_dataset_base = datasets.MNIST('./data', train=True, download=True, transform=transform_no_aug)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform_no_aug)

# ------------------------------------------------------------
print("\n" + "=" * 60)
print("3. 训练无增强模型（基准）")
print("=" * 60)

train_loader_no_aug = DataLoader(train_dataset_base, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

model_no_aug = MNIST_MLP()
criterion = nn.CrossEntropyLoss()
optimizer_no_aug = optim.Adam(model_no_aug.parameters(), lr=0.001)

for epoch in range(1, 4):
    model_no_aug.train()
    for data, target in train_loader_no_aug:
        optimizer_no_aug.zero_grad()
        output = model_no_aug(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_no_aug.step()

# 测试准确率
model_no_aug.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model_no_aug(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
acc_no_aug = 100 * correct / len(test_dataset)
print(f"✅ 无增强模型测试准确率：{acc_no_aug:.2f}%")

# ------------------------------------------------------------
print("\n" + "=" * 60)
print("4. 训练带数据增强模型")
print("=" * 60)

# 用增强版 transform 重新包装训练集
train_dataset_aug = datasets.MNIST('./data', train=True, download=False, transform=transform_with_aug)
train_loader_aug = DataLoader(train_dataset_aug, batch_size=64, shuffle=True)

model_aug = MNIST_MLP()
optimizer_aug = optim.Adam(model_aug.parameters(), lr=0.001)

for epoch in range(1, 4):
    model_aug.train()
    for data, target in train_loader_aug:
        optimizer_aug.zero_grad()
        output = model_aug(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_aug.step()

# 测试准确率
model_aug.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model_aug(data)
        pred = output.argmax(dim=1)
        correct += (pred == target).sum().item()
acc_aug = 100 * correct / len(test_dataset)
print(f"✅ 增强模型测试准确率：{acc_aug:.2f}%")

# ------------------------------------------------------------
print("\n" + "=" * 60)
print("5. 对比结果 & 保存增强模型")
print("=" * 60)

print(f"无增强模型准确率：{acc_no_aug:.2f}%")
print(f"增强模型准确率：{acc_aug:.2f}%")
print(f"提升：{acc_aug - acc_no_aug:.2f} 个百分点")

# 保存增强模型
torch.save(model_aug.state_dict(), "mnist_model_aug.pth")
print("✅ 增强模型已保存为 mnist_model_aug.pth")

print("\n" + "=" * 60)
print("🎉 Day 11 完成！你学会了数据增强，模型鲁棒性更强！")
print("=" * 60)