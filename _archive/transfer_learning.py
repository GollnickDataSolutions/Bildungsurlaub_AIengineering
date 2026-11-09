#%% Pakete
import kagglehub
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision
from PIL import Image

#%% Hyperparameter
EPOCHS = 100
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_TEST = 1000
LEARNING_RATE = 0.001
COLOR_CHANNELS = 3
HORIZONTAL_PIXELS = 32
VERTICAL_PIXELS = 32
# Download latest version
path = "../D03_ComputerVision/MulticlassClassification"

print("Path to dataset files:", path)
# %%
path_train = f"{path}/train"
path_test = f"{path}/test"

#%% Steht eine GPU fürs Training zur Verfügung
device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
# %% Transformationen
train_transforms = transforms.Compose([
    transforms.Resize((HORIZONTAL_PIXELS, VERTICAL_PIXELS)),
    # transforms.Grayscale(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transforms = transforms.Compose([
    transforms.Resize((HORIZONTAL_PIXELS, VERTICAL_PIXELS)),
    # transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


#%% Dataset
train_dataset = torchvision.datasets.ImageFolder(
    root=path_train, 
    transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(
    root=path_test, 
    transform=test_transforms)

#%%
train_dataset.classes

#%%
train_dataset.class_to_idx
# %% DataLoader
train_loader = DataLoader(
    dataset=train_dataset, 
    batch_size=BATCH_SIZE_TRAIN,
    shuffle=True)

test_loader = DataLoader(
    dataset=test_dataset, 
    batch_size=BATCH_SIZE_TEST,
    shuffle=True)


#%% Modellklasse laden (Densenet mit Gewichten)
from torchvision import models
model = models.densenet121(weights=models.DenseNet121_Weights).to(device)

#%% wieviele Parameter hat das Modell?
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters in the model: {num_params}")

#%% Modellgewichte einfrieren
for param in model.parameters():
    param.requires_grad = False

#%% Ausgabeschicht anpassen mit Compose
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(1024, len(train_dataset.classes))
)

#%% Anzahl der trainierbaren Parameter
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters in the model: {num_trainable_params}")



#%% Optimierer und Verlustfunktion
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

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
        loss_train_batch = loss_fn(y_train_pred_batch, y_train_batch)

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


#%% Modell evaluieren auf den Testdaten
with torch.no_grad():
    for (X_test_batch, y_test_batch) in test_loader:
        # 0. Daten auf die Grafikkarte kopieren
        X_test_batch = X_test_batch.to(device)
        y_test_batch = y_test_batch.to(device)
        # 1. Vorhersage von Rohwahrscheinlichkeiten erstellen (logits) [-inf bis +inf]
        y_test_pred_logits = model(X_test_batch)
        # 2. logits --> Wahrscheinlichkeiten [0 bis 1]
        y_test_pred_probs = torch.softmax(y_test_pred_logits, dim=1)
        # 3. Wahrscheinlichkeiten --> Klassen [0 oder 1]
        y_test_pred_classes =  torch.argmax(y_test_pred_probs, dim=1)
y_test_pred_classes

# %% Daten auf CPU kopieren und in Numpy Array umwandeln

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_true=y_test_batch, y_pred=y_test_pred_classes)
sns.heatmap(cm, annot=True, fmt="d")
# %%
accuracy_score(y_true=y_test_batch, y_pred=y_test_pred_classes)
# %% Welche klasse im test_loader ist am häufigsten?
from collections import Counter
Counter(test_loader.dataset.targets)[0]/len(test_loader.dataset.targets)

# %%
