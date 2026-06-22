#%% packages
import kagglehub
import torch
import os
import torchvision
from torchvision import transforms

# %% Hyperparameter
BATCH_SIZE = 64
EPOCHS = 30
LEARNING_RATE = 0.001


#%% Download latest version
path = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")

print("Path to dataset files:", path)

#%% Preprocessing steps
train_transforms = transforms.Compose([
    transforms.RandomVerticalFlip(),
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# (BS, 1, 32, 32)

#%% Dataset Instanzen
train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path, "train"), transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path, "test"), transform=test_transforms)
# %% Dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)

# %% Modellklasse
class ImageClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=16*6*6, out_features=64)
        self.fc2 = torch.nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.conv1(x)  # out: [BS, 6, 30, 30]
        x = self.relu(x)  # aktiviert, aber ändert nichts an der Struktur
        x = self.pool(x) # [BS, 6, 15, 15]
        x = self.conv2(x) # [BS, 16, 13, 13]
        x = self.relu(x)
        x = self.pool(x) # [BS, 16, 6, 6]
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # out: [BS, 1]

model = ImageClassificationModel()
dummy_input = torch.randn((1, 1, 32, 32))
model(dummy_input).shape


# %% Verlustfunktion, Optimizer
loss_fn = torch.nn.BCEWithLogitsLoss()
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
        loss = loss_fn(y_train_pred, y_train_batch.reshape(-1, 1).float())

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
    y_test_pred_batch = torch.sigmoid(y_test_pred_batch)
    y_test_pred_batch = y_test_pred_batch.detach().cpu().flatten().numpy()
    y_test_pred.extend(y_test_pred_batch.tolist())
    y_test.extend(y_test_batch.numpy().tolist())

#%% Umwandlung von Wahrscheinlichkeiten in Klassen
THRESHOLD = 0.5
y_test_pred_classes = [0 if pred<THRESHOLD else 1  for pred in y_test_pred]

# %%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred=y_test_pred_classes, y_true=y_test)
sns.heatmap(cm, annot=True, fmt="d")

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_test_pred_classes)
# %%
import numpy as np
1 - np.sum(y_test) / len(y_test)

# %%
