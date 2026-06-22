#%% Pakete
#%% packages
import torch
import torchvision
from torchvision import transforms, models

from torch.utils.data import random_split
import numpy as np
import os
from collections import OrderedDict

# %% Hyperparameter
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001


#%% Download latest version
path = os.path.join("..","..","data", "tesla_sun_trafficlight")

#%% Preprocessing steps
train_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(),
    # transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    # transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# (BS, 1, 32, 32)

#%% Dataset Instanzen
full_dataset = torchvision.datasets.ImageFolder(root=path, transform=train_transforms)
train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

# %% Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


# %% Modellklasse
model = models.resnet101(pretrained=True)


# dummy_input = torch.randn((1, 3, 32, 32))
# model(dummy_input).shape

#%% Alle Modellgewichte werden eingefroren
for params in model.parameters():
    params.requires_grad = False

# %%
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

#%% Update des Output-Layers
model.fc = torch.nn.Sequential(OrderedDict([
    ('fc1', torch.nn.Linear(in_features=2048, out_features=4))
]))

#%% Count trainable and frozen parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")
print(f"Number of frozen parameters: {frozen_params}")

# %% Verlustfunktion, Optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# %%
train_losses = []
for epoch in range(EPOCHS):
    loss_train_epoch = 0
    for batch, (X_train_batch, y_train_batch) in enumerate(train_loader):
        # gradient reset
        optimizer.zero_grad()

        # forward pass (calculate preds)
        y_train_pred = model(X_train_batch)

        # loss calc
        loss = loss_fn(y_train_pred, y_train_batch)

        # backward pass (calculate gradients)
        loss.backward()

        # weight update
        optimizer.step()

        # store losses
        loss_train_epoch += loss.item()
    train_losses.append(loss_train_epoch)
    print(f"Epoch {epoch}, Loss: {loss_train_epoch}")
# %%
import seaborn as sns
sns.lineplot(train_losses)
# %%
y_test, y_test_pred = [], []
for batch, (X_test_batch, y_test_batch) in enumerate(test_loader):
    y_test_pred_batch = model(X_test_batch)
    y_test_pred_batch = torch.softmax(y_test_pred_batch, dim=1)
    # y_test_pred_batch = y_test_pred_batch.detach().cpu().flatten().numpy()
    y_test_pred.extend(y_test_pred_batch.tolist())
    y_test.extend(y_test_batch.numpy().tolist())

#%% Umwandlung von Wahrscheinlichkeiten in Klassen
y_test_pred_classes = [int(np.argmax(pred)) for pred in y_test_pred]

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred=y_test_pred_classes, y_true=y_test)
sns.heatmap(cm, annot=True, fmt="d")

# save the heatmap
from matplotlib import pyplot as plt
plt.savefig("confusion_matrix.png")

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred_classes)

# print accuracy score
print(f"Accuracy: {accuracy_score(y_test, y_test_pred_classes)}")

# %%
import numpy as np
1 - np.sum(y_test) / len(y_test)

# %%

 