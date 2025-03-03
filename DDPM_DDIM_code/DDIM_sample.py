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
from DDPM_network import *
import numpy as np 
    

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

timesteps = 500

# define beta schedule
betas = linear_beta_schedule(timesteps=timesteps)

# define alphas 
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# calculations for diffusion q(x_t | x_{t-1}) and others
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# calculations for posterior q(x_{t-1} | x_t, x_0)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

ddim_timesteps = 20
c = timesteps // ddim_timesteps
ddim_timesteps_seq =  np.asarray(list(range(0, timesteps, c)))
ddim_timesteps_seq = ddim_timesteps_seq + 1
ddim_timesteps_prev_seq = np.append(np.array([0]), ddim_timesteps_seq[:-1])
ddim_eta = 0
clip_denoised = True



@torch.no_grad()
def p_sample(model, x, t, t_prev):

     # 1. get current and previous alpha_cumprod
    alphas_cumprod_t = extract(alphas_cumprod, t, x.shape)
    alphas_cumprod_t_prev = extract(alphas_cumprod, t_index, x.shape)

     # 2. predict noise using model
    pred_noise = model(x, t)

    pred_x0 = (x - torch.sqrt((1. - alphas_cumprod_t)) * pred_noise) / torch.sqrt(alphas_cumprod_t)

    if clip_denoised:
        pred_x0 = torch.clamp(pred_x0, min=-1., max=1.) # 裁剪梯度

    sigmas_t = ddim_eta * torch.sqrt(
                (1 - alphas_cumprod_t_prev) / (1 - alphas_cumprod_t) * (1 - alphas_cumprod_t / alphas_cumprod_t_prev))
            
    # 5. compute "direction pointing to x_t" of formula (12)
    pred_dir_xt = torch.sqrt(1 - alphas_cumprod_t_prev - sigmas_t**2) * pred_noise

    # 6. compute x_{t-1} of formula (12)
    x_prev = torch.sqrt(alphas_cumprod_t_prev) * pred_x0 + pred_dir_xt + sigmas_t * torch.randn_like(x)
    

    sample_img = x_prev

    return sample_img

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, ddim_timesteps)), desc='sampling loop time step', total=ddim_timesteps):

        img = p_sample(model, img, torch.full((b,), ddim_timesteps_seq[i], device=device, dtype=torch.long), torch.full((b,), ddim_timesteps_prev_seq[i], device=device, dtype=torch.long))
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))