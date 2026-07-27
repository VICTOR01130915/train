# Day 5: PyTorch 张量入门
import torch

print("=" * 40)
print("1. 创建张量")
print("=" * 40)

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
print("张量 a:")
print(a)
print(f"形状: {a.shape}")
print(f"数据类型: {a.dtype}")

b = torch.zeros(2, 3)
c = torch.ones(2, 3)
d = torch.rand(2, 3)
print("\n全0矩阵:\n", b)
print("全1矩阵:\n", c)
print("随机矩阵:\n", d)

print("\n" + "=" * 40)
print("2. 张量运算")
print("=" * 40)

x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])
print(f"x + y = {x + y}")
print(f"x * y = {x * y}")
print(f"x @ y = {x @ y}")
print(f"x.sum() = {x.sum()}")

print("\n" + "=" * 40)
print("3. 张量变形")
print("=" * 40)

matrix = torch.arange(12).reshape(3, 4)
print("3x4 矩阵:\n", matrix)
print("展平后:", matrix.flatten())

print("\nDay 5 完成！")