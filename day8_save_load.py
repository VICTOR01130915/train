# Day 8: 保存与加载模型
# 训练一个简单模型 → 保存到文件 → 加载并预测

import torch
import torch.nn as nn
import torch.optim as optim
import os

print("=" * 50)
print("1. 定义一个简单的分类模型")
print("=" * 50)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
print(model)


print("\n" + "=" * 50)
print("2. 用假数据训练几轮（模拟训练过程）")
print("=" * 50)

# 生成假数据
X = torch.randn(200, 10)
y = torch.randint(0, 2, (200,))

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 6):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


print("\n" + "=" * 50)
print("3. 保存模型")
print("=" * 50)

# 方法一：保存整个模型（推荐方式：保存 state_dict）
model_path = "my_model.pth"
torch.save(model.state_dict(), model_path)
print(f"模型已保存到: {model_path}")
print(f"文件大小: {os.path.getsize(model_path)} 字节")


print("\n" + "=" * 50)
print("4. 重新创建一个新模型（初始参数是随机的）")
print("=" * 50)

new_model = SimpleModel()
print("新模型（未加载前）:")
print(f"fc1.weight 前5个值: {new_model.fc1.weight.data.flatten()[:5]}")


print("\n" + "=" * 50)
print("5. 加载保存的模型参数")
print("=" * 50)

new_model.load_state_dict(torch.load(model_path))
print("加载完成！")
print(f"fc1.weight 前5个值: {new_model.fc1.weight.data.flatten()[:5]}")
print("✅ 参数已恢复为训练后的值！")


print("\n" + "=" * 50)
print("6. 用加载的模型做预测")
print("=" * 50)

# 模拟一条新数据
new_data = torch.randn(1, 10)
new_model.eval()
with torch.no_grad():
    output = new_model(new_data)
    prediction = torch.argmax(output, dim=1)
    print(f"新数据: {new_data[0][:5].tolist()}...")
    print(f"预测类别: {prediction.item()} (0或1)")
    print(f"各类别概率: {torch.softmax(output, dim=1).tolist()}")


print("\n" + "=" * 50)
print("✅ Day 8 完成！你已学会保存和加载模型")
print("=" * 50)
print("\n💡 以后训练模型后，只需保存一次，随时加载使用！")