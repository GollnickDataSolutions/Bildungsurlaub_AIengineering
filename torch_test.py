#%%
import os
import torch
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))
from diffusers import AutoPipelineForText2Image
# %%
pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16",
)
pipe.to("cuda")
#%%
image = pipe(
    prompt="Ein Alpaka steht auf zwei Beinen und schaut in die Kamera, "
           "weiches Studiolicht, fotorealistisch",
    num_inference_steps=1,
    guidance_scale=0.0,
    height=512, width=512,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
#%%
image.save("output.png")

# %% show the image output.png
from IPython.display import Image
Image(filename="output.png")

# %%
