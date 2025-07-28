import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载图像
img = Image.open(r"D:\dataset\ellipse\Prasad\Prasad\images\043_0001.jpg").convert('RGB')  # PIL默认是RGB

# 转换为tensor (C, H, W)，像素值归一化到[0,1]
transform = transforms.ToTensor()
img_tensor = transform(img)

# 打印维度信息
print("Tensor shape:", img_tensor.shape)  # (3, H, W)

# 分离出RGB三个通道
r = img_tensor[0, :, :]
g = img_tensor[1, :, :]
b = img_tensor[2, :, :]

# 可视化
fig, axs = plt.subplots(1, 4, figsize=(15, 5))

# 原图（转回HWC格式）
axs[0].imshow(img_tensor.permute(1, 2, 0))  # C, H, W -> H, W, C
axs[0].set_title("Original RGB Image")

# 单通道图
axs[1].imshow(r, cmap='Reds')
axs[1].set_title("Red Channel")

axs[2].imshow(g, cmap='Greens')
axs[2].set_title("Green Channel")

axs[3].imshow(b, cmap='Blues')
axs[3].set_title("Blue Channel")

for ax in axs:
    ax.axis('off')
    ax.set_aspect('equal')   #保持原图的比例
plt.tight_layout()
plt.show()
