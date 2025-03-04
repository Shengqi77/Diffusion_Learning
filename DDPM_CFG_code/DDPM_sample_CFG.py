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

timesteps = 400


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

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)



@torch.no_grad()
def p_sample(model, x, t, t_index, context_mask, timesteps, n_sample, c_i):

    guide_w = 1 # Guidence Coefficient
    t_is = t_index / timesteps
    t_is = t_is.repeat(n_sample)
    # 
    x = x.repeat(2,1,1,1)
    t_is = t_is.repeat(2)
    # split predicted noise(one is context-guided, another is context-guided free)
    eps = model(x, c_i, t_is, context_mask)
    eps1 = eps[:n_sample]
    eps2 = eps[n_sample:]
    eps = (1+guide_w)*eps1 - guide_w*eps2 

    x = x[:n_sample]
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    noise = torch.randn_like(x)
    x = sqrt_recip_alphas_t * (x - eps * betas_t /sqrt_one_minus_alphas_cumprod_t) + torch.sqrt(betas_t) * noise

    return x

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device
    b = shape[0] # bachsize 
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    c_i = torch.arange(0,10).to(device) # context for us just cycles throught the mnist label
    c_i = c_i.repeat(int(b/c_i.shape[0]))
    # don't drop context at test time
    context_mask = torch.zeros_like(c_i).to(device)

    # double the batch for computing the CFG-based result
    c_i = c_i.repeat(2)
    context_mask = context_mask.repeat(2)
    context_mask[b:] = 1. # make second half of batch context free
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long) ,torch.tensor(i).to(device), context_mask, timesteps, b, c_i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=3):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))