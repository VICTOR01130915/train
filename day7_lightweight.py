# Day 7: 轻量版神经网络
import torch
import torch.nn as nn
import torch.optim as optim

print("=" * 50)
print("1. 定义网络结构")
print("=" * 50)

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleMLP()
print(model)

print("\n" + "=" * 50)
print("2. 生成假数据（100条样本）")
print("=" * 50)

X = torch.randn(100, 10)
y = torch.randint(0, 2, (100,))
print(f"输入形状: {X.shape}")
print(f"标签形状: {y.shape}")

print("\n" + "=" * 50)
print("3. 训练3轮")
print("=" * 50)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(1, 4):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print("\n" + "=" * 50)
print("✅ Day 7 完成！你已跑通神经网络训练全流程")
print("=" * 50)
