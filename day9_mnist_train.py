# Day 9: MNIST 手写数字识别 - 真实数据集训练

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

print("=" * 60)
print("1. 准备 MNIST 数据集")
print("=" * 60)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集: {len(train_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")


print("\n" + "=" * 60)
print("2. 定义网络结构")
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

model = MNIST_MLP()
print(model)


print("\n" + "=" * 60)
print("3. 定义损失函数和优化器")
print("=" * 60)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


print("\n" + "=" * 60)
print("4. 开始训练（3轮）")
print("=" * 60)

epochs = 3
train_losses = []
train_accs = []

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    train_losses.append(avg_loss)
    train_accs.append(train_acc)

    print(f"Epoch {epoch} 平均Loss: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%")

torch.save({'losses': train_losses, 'accs': train_accs}, 'mlp_training_log.pth')
print("✅ MLP 训练记录已保存")


print("\n" + "=" * 60)
print("5. 测试集评估")
print("=" * 60)

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

accuracy = 100 * correct / total
print(f"🎯 测试集准确率: {accuracy:.2f}%")

print("\n" + "=" * 60)
print("🎉 Day 9 完成！")
print("=" * 60)