from pathlib import Path
import matplotlib.animation as animation
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
import torch
from tqdm.auto import tqdm
import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
from DDPM_network_CFG import *
from DDPM_sample_CFG import *


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# forward diffusion (using the nice property)
def q_sample(x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def p_losses(denoise_model, x_start, x_class, t, timesteps, noise=None, loss_type="l1", device = None):
    # 先采样噪声
    if noise is None:
        noise = torch.randn_like(x_start)
    
    # dropout context with some probability
    drop_prob = 0.1
    context_mask = torch.bernoulli(torch.zeros_like(x_class)+drop_prob).to(device)

    # 用采样得到的噪声去加噪图片
    x_noisy = q_sample(x_start=x_start, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, x_class, t/ timesteps, context_mask)
    
    # 根据加噪了的图片去预测采样的噪声
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss