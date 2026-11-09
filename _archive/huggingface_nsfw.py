
#%% Use a pipeline as a high-level helper
from PIL import Image
from transformers import pipeline
#%%
img = Image.open("waldo.jpg")
classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")
classifier(img)

# %%
