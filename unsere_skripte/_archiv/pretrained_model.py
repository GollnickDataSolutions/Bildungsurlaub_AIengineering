#%% Pakete
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import kagglehub
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from datetime import datetime
# %% Download latest version
target_path = os.path.expanduser("~/.cache/kagglehub/datasets/samuelcortinhas/muffin-vs-chihuahua-image-classification/versions/2")
if not os.path.exists(target_path):
    path = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")
else:
    path = target_path

#%% Hyperparameter
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
DEVICE = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = 224
#%%
print("Path to dataset files:", path)
test_folder = os.path.join(path, "test")
train_folder = os.path.join(path, "train")

#%% Dataset Instanzen erstellen
train_preprocessing_steps = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    # transforms.Grayscale(),
    transforms.ToTensor()
])

test_preprocessing_steps = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    # transforms.Grayscale(),
    transforms.ToTensor()
])

test_dataset = ImageFolder(root=test_folder, transform=train_preprocessing_steps)
train_dataset = ImageFolder(root=train_folder, transform=test_preprocessing_steps)

# %% DataLoader erstellen
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

#%% Modellklasse erstellen
from torchvision import models
model = models.resnet18(pretrained=True)

#%% Modellgewichte einfrieren
for params in model.parameters():
    params.requires_grad = False  

#%%
model.fc = nn.Linear(512, 1)


# %% Anzahl der Modellparameter
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of model parameters: {total_params}")
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

#%% Optimierer und Verlustfunktion
loss_fn = torch.nn.BCEWithLogitsLoss()
#%% Optimierer festlegen
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
 
# %% Trainingsschleife
#%% Trainingsschleife
time_start = datetime.now()

losses_train, losses_val = [], []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
        model.train()

        # Trainingsdaten auf GPU kopieren
        X_train_batch, y_train_batch = X_train_batch.to(DEVICE), y_train_batch.to(DEVICE)
        # Gradienten nullen
        optimizer.zero_grad()

        # Forward Pass
        y_train_batch_pred = model(X_train_batch)

        # Verluste berechnen
        # Ensure both tensors are float32 and have the same shape for the loss calculation
        loss = loss_fn(y_train_batch_pred, y_train_batch.reshape(-1, 1).float())
        loss_epoch += loss.item()

        # Backward Pass
        loss.backward()

        # Parameter updaten
        optimizer.step()

    # Verluste speichern für spätere Visualisierung
    losses_train.append(loss_epoch/len(train_loader))

    print(f"Epoche {epoch}, Train-Verlust {loss}")
time_end = datetime.now()
print(f"Das Training hat {time_end - time_start} gedauert.")

# %% Visualisierung der Trainingsverluste
import seaborn as sns
sns.lineplot(losses_train)

#%%
model.eval()
y_test = []
y_test_pred = []
with torch.no_grad():
    for (X_test_batch, y_test_batch) in test_loader:
        y_test_batch_logit = model(X_test_batch.to(DEVICE))
        y_test.extend(y_test_batch.cpu().numpy().tolist())
        y_test_pred.extend(torch.sigmoid(y_test_batch_logit).cpu().numpy().flatten().tolist())
# %%
THRESHOLD = 0.5
y_test_pred_classes = [0 if pred<THRESHOLD else 1 for pred in y_test_pred]
# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred_classes)
sns.heatmap(cm, annot=True, fmt="d")
# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_true=y_test, y_pred=y_test_pred_classes)

# %%
