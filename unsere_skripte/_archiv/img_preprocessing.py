#%% pakete
from torchvision import transforms
from PIL import Image

#%%
img = Image.open('waldo.jpg')
img

#%%
preprocessing_steps = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(),
    transforms.ToTensor()
    
])
preprocessing_steps(img).shape
# %%
