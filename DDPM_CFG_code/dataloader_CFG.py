from datasets import load_dataset
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

batch_size = 256



tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

dataset = MNIST("./data", train=True, download=True, transform=tf)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=5)


