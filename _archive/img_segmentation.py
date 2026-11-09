#%%
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-sem.yaml")  # build a new model from YAML
model = YOLO("yolo26n-sem.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n-sem.yaml").load("yolo26n-sem.pt")  # build from YAML and transfer weights

# Train the model
results = model.train(data="cityscapes8.yaml", epochs=100, imgsz=1024)
# %%
results
# %%
