#%% Pakete
import kagglehub
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from PIL import Image

#%% Hyperparameter
EPOCHS = 10
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.001
COLOR_CHANNELS = 1
HORIZONTAL_PIXELS = 32
VERTICAL_PIXELS = 32
# Download latest version
path = kagglehub.dataset_download("samuelcortinhas/muffin-vs-chihuahua-image-classification")

print("Path to dataset files:", path)
# %%
path_train = f"{path}/train"
path_test = f"{path}/test"

#%% Steht eine GPU fürs Training zur Verfügung
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %% Transformationen
train_transforms = transforms.Compose([
    transforms.Resize((HORIZONTAL_PIXELS, VERTICAL_PIXELS)),
    transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((HORIZONTAL_PIXELS, VERTICAL_PIXELS)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


#%% Dataset
train_dataset = torchvision.datasets.ImageFolder(
    root=path_train, 
    transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(
    root=path_test, 
    transform=test_transforms)
# %% DataLoader
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE_TEST,
    shuffle=True)

#%% Modellklasse erstellen
class ImageClassificationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=COLOR_CHANNELS, out_channels=6, kernel_size=3)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
        self.flatten = torch.nn.Flatten()
        self.fc_out = torch.nn.Linear(576, 1)
        # TODO: 576 ist noch hardgecodet, sollte später dynamisch abgeleitet werden.

    def forward(self, x):
        x = self.conv1(x)  # (BATCH_SIZE, 6, 30, 30)
        x = self.relu(x)   # nur Aktivierung, keine Größenänderung
        x = self.pool(x)   # (BATCH_SIZE, 6, 15, 15)
        x = self.conv2(x)  # (BATCH_SIZE, 16, 13, 13)
        x = self.relu(x)   # nur Aktivierung, keine Größenänderung
        x = self.pool(x)   # (BATCH_SIZE, 16, 6, 6)
        x = self.flatten(x) # (BATCH_SIZE, 576)
        x = self.fc_out(x)  # (BATCH_SIZE, 1)
        return x

# Modellinstanz erstellen
# dummy_input = torch.randn(BATCH_SIZE_TRAIN, COLOR_CHANNELS, HORIZONTAL_PIXELS, VERTICAL_PIXELS)
model = ImageClassificationModel().to(device=device)
# model(dummy_input).shape

#%% Optimierer und Verlustfunktion
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.BCEWithLogitsLoss()

#%% Modelltraining durchführen
losses_train = []

for epoch in range(EPOCHS):
    loss_train_epoch = 0
    for (X_train_batch, y_train_batch) in train_loader:
        # 0. Daten auf die Grafikkarte kopieren
        X_train_batch = X_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)

        # 1. Setze Gradienten auf Null
        optimizer.zero_grad()

        # 2. Forward-Pass (Berechnung der Vorhersagen)
        y_train_pred_batch = model(X_train_batch)

        # 3. Verluste ermitteln
        loss_train_batch = loss_fn(y_train_pred_batch.flatten(), y_train_batch.float())

        # 4. Gradienten berechnen
        loss_train_batch.backward()

        # 5. Modellgewichte anpassen
        optimizer.step()

        # Trainings-Verluste wegspeichern (optional) and normalize by batch size
        loss_train_epoch += loss_train_batch.item() / len(train_loader)
    losses_train.append(loss_train_epoch)
    print(f"Epoch {epoch}, Loss: {loss_train_epoch}")
    
#%% Plot training loss
import seaborn as sns
sns.lineplot(x=range(EPOCHS), y=losses_train)

#%% Dummy-Classifier (Baseline)
# get the number of classes
import numpy as np
print(f"Baseline Accuracy: {(1 - np.sum(test_loader.dataset.targets) / len(test_loader.dataset.targets))*100} %")


#%% Modell evaluieren auf den Testdaten
THRESHOLD = 0.5
with torch.no_grad():
    for (X_test_batch, y_test_batch) in test_loader:
        # 0. Daten auf die Grafikkarte kopieren
        X_test_batch = X_test_batch.to(device)
        y_test_batch = y_test_batch.to(device)
        # 1. Vorhersage von Rohwahrscheinlichkeiten erstellen (logits) [-inf bis +inf]
        y_test_pred_logits = model(X_test_batch)
        # 2. logits --> Wahrscheinlichkeiten [0 bis 1]
        y_test_pred_probs = torch.sigmoid(y_test_pred_logits).flatten()
        # 3. Wahrscheinlichkeiten --> Klassen [0 oder 1]
        y_test_pred_classes = (y_test_pred_probs >= THRESHOLD).float()
y_test_pred_classes

# %%
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true=y_test_batch, y_pred=y_test_pred_classes)
sns.heatmap(cm, annot=True, fmt="d")
# %%
accuracy_score(y_true=y_test_batch, y_pred=y_test_pred_classes)
# %%
