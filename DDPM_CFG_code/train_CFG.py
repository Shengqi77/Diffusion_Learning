from pathlib import Path
import matplotlib.animation as animation
from DDPM_network_CFG import Unet
from dataloader_CFG import *
from losses_forward import *
from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
import torch
from tqdm.auto import tqdm
import math
from inspect import isfunction
from functools import partial
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from einops import rearrange
from torch import nn, einsum
import torch.nn.functional as F
import os
from DDPM_sample_CFG import *


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

results_folder = Path("./results")
results_folder.mkdir(exist_ok = True)
save_and_sample_every = 1000
image_size = 28
channels = 1

from torch.optim import Adam

device = "cuda" if torch.cuda.is_available() else "cpu"

model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

from torchvision.utils import save_image

epochs = 100
timesteps = 400

model_save_folder = results_folder / "saved_models"
os.makedirs(model_save_folder, exist_ok=True)

for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        batch_size = batch[0].shape[0]
        batch_pixel = batch[0].to(device)
        batch_class = batch[1].to(device)


        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        loss = p_losses(model, batch_pixel, batch_class, t, timesteps, loss_type="huber", device=device)

        if step % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(dataloader)}], Loss: {loss.item()}")

        loss.backward()
        optimizer.step()
    # 在每个 epoch 结束时保存模型
    model_save_path = model_save_folder / f"model_epoch_{epoch+1}.pth"
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, model_save_path)
    print(f"Model saved to {model_save_path}")

