# Day 16: 画训练曲线图
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']

print("=" * 50)
print("加载训练记录...")
print("=" * 50)

mlp = torch.load('mlp_training_log.pth')
cnn = torch.load('cnn_training_log.pth')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

epochs = range(1, len(mlp['losses']) + 1)

ax1.plot(epochs, mlp['losses'], 'bo-', label='MLP')
ax1.plot(epochs, cnn['losses'], 'ro-', label='CNN')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练 Loss 对比')
ax1.legend()
ax1.grid(True)

ax2.plot(epochs, mlp['accs'], 'bo-', label='MLP')
ax2.plot(epochs, cnn['accs'], 'ro-', label='CNN')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('准确率 (%)')
ax2.set_title('训练准确率对比')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_comparison.png', dpi=200)
print("\n✅ 对比图已保存为 training_comparison.png")
plt.show()