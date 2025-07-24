import torch

class DDPM():

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta,max_beta,n_steps).to(device)
        alphas = 1 - betas
