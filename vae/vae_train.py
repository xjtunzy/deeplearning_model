import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from vae_model import VAE
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(
    root='../data', 
    train=True, 
    download=True, 
    transform=transform
)

#生成器
train_dataloader = DataLoader(train_dataset,batch_size=64,shuffle = True)

#检测是否有可用的gpu
device = torch.device("cuda")
#实例化网络
model = VAE()
model.to(device)
#定义损失函数
def lossf(recon_x, x, mu, logvar):
    #重建损失
    loss1 = F.mse_loss(recon_x,x.view(x.size(0),-1),reduction='sum')
    #KL散度损失
    loss2 = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return loss1+loss2
#定义优化器
optimizer = optim.Adam(model.parameters(), lr=1e-3)

#开始训练
train_losses = []
for epoch in range(50):
    model.train()
    train_loss = 0
    for img,_ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        data = img.to(device)
        # 前向传播
        recon_batch, mu, logvar = model(data)
        
        # 计算损失
        loss = lossf(recon_batch, data, mu, logvar)
        
        # 反向传播与优化
        optimizer.zero_grad()  #把之前的梯度置为0
        loss.backward()  #自动求导，实现梯度的反向传播
        optimizer.step() #更新参数
        
        train_loss += loss.item()
    print(f"train_loss: {train_loss}")
    train_losses.append(train_loss)
torch.save(model.state_dict(),"../model_weight\\vae_model.pth")
#画出训练损失
plt.title("train_loss")
plt.plot(np.arange(len(train_losses)),train_losses)
plt.legend(['train_loss'],loc = 'upper right')
plt.show()
