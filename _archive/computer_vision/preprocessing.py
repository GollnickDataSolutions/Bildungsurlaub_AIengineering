#%% Pakete
import torch
from PIL import Image
from torchvision import transforms

# %%
img = Image.open("kiki.jpg")
img
# %%
preprocess_steps = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.CenterCrop(300),
    transforms.Grayscale(),
    # transforms.RandomRotation(30),
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])
preprocess_steps(img).shape

# %%
