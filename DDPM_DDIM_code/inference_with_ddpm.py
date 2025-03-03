from pathlib import Path
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ddpm import *
device = "cuda" if torch.cuda.is_available() else "cpu"
# 加载训练好的模型
model_save_folder = Path("./results/saved_models")
model_save_path = model_save_folder / "model_epoch_5.pth"  # 替换为你保存的模型文件名

image_size = 28
channels = 1
# 定义模型（确保与训练时的模型结构一致）
model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1, 2, 4,)
)
model.to(device)

# 加载模型状态
checkpoint = torch.load(model_save_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 设置为评估模式


# 生成并保存 GIF
random_index = 53

fig = plt.figure()
ims = []
with torch.no_grad():  # 不计算梯度
    # sample 64 images
    samples = sample(model, image_size=image_size, batch_size=64, channels=channels)

for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')






