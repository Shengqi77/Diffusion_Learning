from pathlib import Path
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from DDPM_network_CFG import *
from DDPM_sample_CFG import *
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model_save_folder = Path("./results/saved_models")
model_save_path = model_save_folder / "model_epoch_82.pth" 
image_size = 28
channels = 1
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)
checkpoint = torch.load(model_save_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  

fig, ax = plt.subplots()
ax.axis("off")  
ims = []

batch_size = 40  
with torch.no_grad():  
    # sample 40 images
    samples = sample(model, image_size=image_size, batch_size=batch_size, channels=channels)

for i in range(0, timesteps + 50, 50):  
    if i == timesteps:
        i = i - 1 
    batch_images = samples[i]  
    batch_images = batch_images.reshape(batch_size, image_size, image_size, channels)# (40, image_size, image_size, channels)
    if batch_images.shape[-1] == 1:
        batch_images = batch_images.squeeze(-1)   #(40, image_size, image_size)
    grid_rows = []
    for j in range(0, 40, 10):  
        row = np.hstack(batch_images[j:j+10])  
        grid_rows.append(row)
    grid_image = np.vstack(grid_rows) 
    im = ax.imshow(grid_image, cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=1000)
animate.save('diffusion_grid.gif', writer='pillow')
plt.show()























# for i in range(0,timesteps,50):
#     im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
#     ims.append([im])

# animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
# animate.save('diffusion.gif')
# plt.show()




