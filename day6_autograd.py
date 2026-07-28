#day6:Pytorch 自动求导
#这是神经网络训练的"发动机"

import torch

print("="*50)
print("1. 基础求导:y=x² 在x=2处的导数")
print("="*50)

#创建一个张量，requires_grad=True 表示需要计算梯度
x= torch.tensor([2.0],requires_grad=True)
print(f"x={x}")

#定义函数y=x²
y=x ** 2
print(f"y=x²={y}")

#反向传播,计算梯度
y.backward()
print(f"dy/dx={x.grad}")
print("数学上:d(x²)/dx=2x,在x=2时=4√")

print("\n" + "="*50)
print("2.多变量求导:z=x²+y³")
print("="*50)

x=torch.tensor([2.0],requires_grad=True)
y=torch.tensor([3.0],requires_grad=True)
z=x**2+y**3
print(f"x={x},y={y}")
print(f"z= x²+y³")

#反向传播
z.backward()
print(f"∂z/∂x={x.grad}") #2*x=4
print(f"∂z/∂y={y.grad}") #3*y²=27

print("\n"+"="*50)
print("3.矩阵求导(神经网络最常用)")
print("="*50)

W=torch.tensor([[2.0,3.0]],requires_grad=True) #权重矩阵1x2
x=torch.tensor([[4.0],[5.0]])                  #输入2x1
b=torch.tensor([1.0],requires_grad=True)

y=W @ x +b  #矩阵乘法 +偏置
print(f"w={W}")
print(f"x={x}")
print(f"b={b}")
print(f"y=wx+b={y}")

#假设损失函数L=y²,求损失W和b的梯度
loss=y**2
loss.backward()
print(f"\n∂L/∂W={W.grad}")
print(f"∂L\∂b={b.grad}")

print("\n" + "=" *50)
print("4. 梯度累计与清零(重要!)")
print("=" *50)

x=torch.tensor([2.0],requires_grad=True)
y=x**2
y.backward()
print(f"第一次求导后,x.grad={x.grad}")

#如果不手动清零，梯度会累积！
y2=x**3
y2.backward()
print(f"第二次求导后(未清零),x.grad={x.grad}")

#手动清零
x.grad.zero_()
print(f"清零后,x.grad={x.grad}")

y3 = x**4
y3.backward()
print(f"清零后再求导,x.grad={x.grad}")  #d(x^4)/dx=4*x³=32

print("\n" + "="*50)
print("5.禁用梯度运算(推理/评估时用)")
print("="*50)

x=torch.tensor([3.0],requires_grad=True)
print(f"x={x}, requires_grad={x.requires_grad}")

#用torch.no_grad():
with torch.no_grad():
    y=x**2
    print(f"在no_grad下计算y=x²={y}")
    print(f"此时y.requires_grad={y.requires_grad} (False,不追踪梯度)")

print("\n"+"="*50)
print("√ day6完成!")
print("="*50)