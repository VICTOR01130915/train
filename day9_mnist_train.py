# Day 9: MNIST 手写数字识别 - 真实数据集训练
# 你人生中第一个真正有用的模型！

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

print("=" * 60)
print("1. 准备 MNIST 数据集（首次运行会自动下载）")
print("=" * 60)

# 数据预处理：转为张量 + 归一化
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 下载训练集和测试集
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集: {len(train_dataset)} 张图片")
print(f"测试集: {len(test_dataset)} 张图片")
print(f"每批大小: 64")


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
print("4. 开始训练（3轮，约2-3分钟）")
print("=" * 60)

epochs = 3

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

        if batch_idx % 300 == 0:
            print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}], "
                  f"Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total
    print(f"Epoch {epoch} 平均Loss: {avg_loss:.4f}, 训练准确率: {train_acc:.2f}%")


print("\n" + "=" * 60)
print("5. 在测试集上评估")
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
print("6. 保存模型")
print("=" * 60)

model_path = "mnist_model.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ 模型已保存到: {model_path}")
print(f"文件大小: {os.path.getsize(model_path)} 字节")


print("\n" + "=" * 60)
print("7. 展示几张图片的预测结果")
print("=" * 60)

import matplotlib.pyplot as plt

dataiter = iter(test_loader)
images, labels = next(dataiter)

fig, axes = plt.subplots(1, 8, figsize=(12, 3))
model.eval()
with torch.no_grad():
    outputs = model(images[:8])
    _, predicted = torch.max(outputs, 1)

for i in range(8):
    img = images[i].squeeze().numpy()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f'Pred: {predicted[i].item()}\nTrue: {labels[i].item()}')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('mnist_predictions.png')
print("✅ 预测结果图已保存: mnist_predictions.png")


print("\n" + "=" * 60)
print("🎉 Day 9 完成！你拥有了第一个真实可用的模型！")
print(f"   准确率: {accuracy:.2f}%")
print("=" * 60)