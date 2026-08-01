# Day 10: 用训练好的模型识别我自己手写的数字
# 修复版：自动反转颜色，匹配MNIST训练数据格式

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import os

print("=" * 60)
print("1. 定义网络结构（必须和训练时完全一致）")
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


print("\n" + "=" * 60)
print("2. 加载训练好的模型")
print("=" * 60)

model = MNIST_MLP()
model_path = "mnist_model.pth"

if not os.path.exists(model_path):
    print(f"❌ 找不到模型文件: {model_path}")
    print("请先运行 day9_mnist_train.py 训练模型")
    exit()

model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
print("✅ 模型加载成功！")


print("\n" + "=" * 60)
print("3. 加载并预处理你手写的数字图片")
print("=" * 60)

image_path = "my_digit.png"

if not os.path.exists(image_path):
    print(f"❌ 找不到图片: {image_path}")
    print("请先在画图工具里画一个数字，保存为 my_digit.png")
    exit()

# 加载图片并转换为灰度图
img = Image.open(image_path).convert('L')
print(f"原始图片大小: {img.size}")

# 🔥 关键修复：颜色反转（白底黑字 → 黑底白字，匹配MNIST）
img = ImageOps.invert(img)
print("✅ 已执行颜色反转（白底黑字 → 黑底白字）")

# 预处理：缩放到 28x28，转为张量，归一化
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

img_tensor = transform(img).unsqueeze(0)  # 增加 batch 维度
print(f"预处理后张量形状: {img_tensor.shape}")


print("\n" + "=" * 60)
print("4. 让模型预测！")
print("=" * 60)

with torch.no_grad():
    output = model(img_tensor)
    probabilities = torch.softmax(output, dim=1)
    prediction = torch.argmax(output, dim=1).item()
    confidence = probabilities[0][prediction].item() * 100

print(f"🎯 模型预测结果: {prediction}")
print(f"📊 置信度: {confidence:.2f}%")

# 显示所有类别的概率
print("\n各类别概率:")
for i, prob in enumerate(probabilities[0]):
    print(f"  {i}: {prob.item() * 100:.2f}%")


print("\n" + "=" * 60)
print("5. 显示图片和预测结果")
print("=" * 60)

import matplotlib.pyplot as plt

plt.figure(figsize=(4, 4))
plt.imshow(img, cmap='gray')
plt.title(f"AI 预测结果: {prediction}\n置信度: {confidence:.2f}%")
plt.axis('off')
plt.savefig('my_digit_prediction.png')
print("✅ 结果图已保存: my_digit_prediction.png")


print("\n" + "=" * 60)
print("🎉 Day 10 完成！你已经完成了完整的 AI 项目闭环！")
print(f"   你写的数字 → AI 识别为 → {prediction}")
print("=" * 60)