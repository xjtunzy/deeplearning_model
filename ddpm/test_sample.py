from ddpm_sample import DDPM
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
#检测是否有可用的gpu
device = torch.device("cuda:0")
ddpm = DDPM(device,1000)

idx = []
for i in range(0,1000,199):
    idx.append(i)
print(f"idx: {idx}")

#读入图片
filename = r"D:\dataset\ellipse\Prasad\Prasad\images\043_0001.jpg"
img = Image.open(filename)  # 用 PIL 打开
transform = transforms.ToTensor()  # 会自动归一化
img_tensor = transform(img)  # shape: [3, H, W]
img_tensor = img_tensor.to(device)
img_res =[]
for i in range(len(idx)):
    t = idx[i]
    res = ddpm.sample_forward(img_tensor,t)
    img_res.append(res)

print(f"len of img_res: {len(img_res)}")

fig, axes = plt.subplots(1, 6,figsize=(len(img_res)*2.5, 3))  
for i, ax in enumerate(axes.flat):
    ax.imshow(img_res[i].squeeze(0).permute(1, 2, 0).cpu().numpy())  # 显示图像
    ax.axis('off')  # 不显示坐标轴
    ax.set_title(f"step: {idx[i]}")
    ax.set_aspect('equal')   #保持原图的比例

plt.tight_layout()
plt.savefig('sample_forward_photo\\samples_forward.png')
plt.show()