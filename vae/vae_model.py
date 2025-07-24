import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self,latent_dim=50):
        super(VAE, self).__init__()

        #编码器,mlp
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,400),
            nn.ReLU(),
            nn.Linear(400,300),
            nn.ReLU()
        )
        #均值
        self.mu = nn.Linear(300,latent_dim)
        self.logvar = nn.Linear(300,latent_dim)

        #解码器，mlp
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim,300),
            nn.ReLU(),
            nn.Linear(300,400),
            nn.ReLU(),
            nn.Linear(400,28*28),
            nn.Tanh()
        )

    #重参数技巧
    def get_latent_z(self,mu,logvar):
        var = torch.exp(0.5*logvar)
        eps = torch.randn_like(var)
        return mu + eps*var  #注意*是直接让两个相同tensor对应元素相乘
    
    def forward(self,x):
        l1 = self.encoder(x)
        m = self.mu(l1)
        v = self.logvar(l1)
        z = self.get_latent_z(m,v)
        xt = self.decoder(z)
        return xt,m,v
    
