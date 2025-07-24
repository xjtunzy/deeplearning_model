#测试训练好的vae模型
import torch
from vae_model import VAE
import matplotlib.pyplot as plt
device = torch.device("cuda")
model = VAE().to(device)
model.load_state_dict(torch.load("..\\model_weight\\vae_model.pth", map_location=device, weights_only=True))  # 加载参数
model.eval()                     # 设置为评估模式（推理用）

#使用模型推理
with torch.no_grad():
    # 从标准正态分布采样潜在变量
    z = torch.randn(16, 50).to(device)
    samples = model.decoder(z).cpu().view(-1, 28, 28)

samples = samples.detach().numpy()  # 转换成 numpy 数组以便 plt 使用

fig, axes = plt.subplots(4, 4, figsize=(6, 6))  # 显示 4x4 的图像网格
for i, ax in enumerate(axes.flat):
    ax.imshow(samples[i], cmap='gray')  # 显示灰度图像
    ax.axis('off')  # 不显示坐标轴

plt.tight_layout()
plt.savefig('..\\model_generate_photo\\samples_vae.png')
plt.show()