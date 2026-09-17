#%% Packages
import torch
from torchvision import transforms
from PIL import Image
# %%
img = Image.open("kiki.jpg")
img

#%% Preprocessing Schritte
preprocess_steps = transforms.Compose([
    transforms.Resize(size=(300,300)),
    # transforms.CenterCrop(150),
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(),
    # transforms.ToTensor()

])
preprocess_steps(img)
# %%
