#%% packages
import kagglehub
import torch
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%% Constants
BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VAL = 1000
BATCH_SIZE_TEST = 1000
LR = 0.001
#%% Download latest version
path = kagglehub.dataset_download("developerghost/intrusion-detection-logs-normal-bot-scan")

print("Path to dataset files:", path)
df = pd.read_csv(f"{path}/Network_logs.csv")
df
# %% alle spalten in kleinbuchstaben umwandeln
df.columns = df.columns.str.lower()
df.columns
#%%
df["destination_ip"].nunique()

#%% Spalten löschen
cols_to_delete = ["source_ip", "destination_ip", "scan_type", "port"]
df.drop(columns=cols_to_delete, inplace=True)
df.shape

#%% Behandlung der kathegorischen Daten
# TODO: get_dummies implementieren
categorical_cols = ['request_type', 'protocol', 'user_agent', 'status']
df_dummies = pd.get_dummies(data=df, columns=categorical_cols, dtype=int, drop_first=True)
df_dummies

#%% Alternativ nur Spalte "port" in Categorical umwandeln und dann alle kathegorischen/string Spalten in Dummies umwandeln
# df['port']  = pd.Categorical(df['port'])
df_dummies = pd.get_dummies(data=df, dtype=int, drop_first=True)
df_dummies
# %% X und y trennen
X = df_dummies.drop(columns=["intrusion"])
y = df_dummies["intrusion"]

#%% train / test / validation split
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=BATCH_SIZE_VAL + BATCH_SIZE_TEST, random_state=42, stratify=y)
X_test, X_val, y_test, y_val = train_test_split(X_val_test, y_val_test, test_size=BATCH_SIZE_TEST, random_state=42, stratify=y_val_test)
print("Train set size:", X_train.shape)
print("Test set size:", X_test.shape)
print("Validation set size:", X_val.shape)

#%% Verteilung der Zielgröße
from collections import Counter
Counter(y)

#%% daten skalieren
scaler = StandardScaler()
X_train_scaled = np.array(scaler.fit_transform(X_train), dtype=np.float32)
X_val_scaled = np.array(scaler.transform(X_val), dtype=np.float32)
X_test_scaled = np.array(scaler.transform(X_test), dtype=np.float32)

# %% Dataset and DataLoader
from torch.utils.data import DataLoader, Dataset
class BinaryClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(np.array(y, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = BinaryClassificationDataset(X_train_scaled, y_train)
val_dataset = BinaryClassificationDataset(X_val_scaled, y_val)
test_dataset = BinaryClassificationDataset(X_test_scaled, y_test)

# %% DataLoader erstellen
train_loader = DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE_TRAIN, 
    shuffle=True)
val_loader = DataLoader(
    val_dataset, 
    batch_size=1000, 
    shuffle=False)
test_loader = DataLoader(
    test_dataset, 
    batch_size=1000, 
    shuffle=False)

# %% Modellklasse erstellen
class BinaryClassificationModel(torch.nn.Module):
    def __init__(self, input_dim, hidden1, hidden2):
        super(BinaryClassificationModel, self).__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden1)
        self.layer_2 = torch.nn.Linear(hidden1, hidden2)
        self.output = torch.nn.Linear(hidden2, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.output(x)
        
        return x

INPUT_DIM = X_train_scaled.shape[1]
model = BinaryClassificationModel(
    input_dim=INPUT_DIM, 
    hidden1=16, 
    hidden2=8)

# %% Verlustfunktion und Optimierer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.BCEWithLogitsLoss()

# %% Trainingsschleife
NUM_EPOCHS = 100
losses_train, losses_val = [], []

for epoch in range(NUM_EPOCHS):
    loss_train_epoch = 0
    for (X_train_batch, y_train_batch) in train_loader:
        # 1. Setze Gradienten auf Null
        optimizer.zero_grad()

        # 2. Forward-Pass (Berechnung der Vorhersagen)
        y_train_pred_batch = model(X_train_batch)

        # 3. Verluste ermitteln
        loss_train_batch = loss_fn(y_train_pred_batch.flatten(), y_train_batch)

        # 4. Gradienten berechnen
        loss_train_batch.backward()

        # 5. Modellgewichte anpassen
        optimizer.step()

        # Trainings-Verluste wegspeichern (optional) and normalize by batch size
        loss_train_epoch += loss_train_batch.item() / len(train_loader)
    losses_train.append(loss_train_epoch)
    print(f"Epoch {epoch}, Loss: {loss_train_epoch}")

    # Validation-Verluste ermitteln
    with torch.no_grad():
        for (X_val_batch, y_val_batch) in val_loader:
            y_val_pred = model(X_val_batch)  # Vorhersage erstellen
            loss_val = loss_fn(y_val_pred.flatten(), y_val_batch)  # Verlust ermitteln
            losses_val.append(loss_val.item() / len(val_loader)) # Verlust abspeichern

#%% 
sns.lineplot(data=[losses_train[20: ], losses_val[20:]])

#%% Evaluierung der Test-Daten
THRESHOLD = 0.5
with torch.no_grad():
    for (X_test_batch, y_test_batch) in test_loader:
        # 1. Vorhersage von Rohwahrscheinlichkeiten erstellen (logits) [-inf bis +inf]
        y_test_pred_logits = model(X_test_batch)
        # 2. logits --> Wahrscheinlichkeiten [0 bis 1]
        y_test_pred_probs = torch.sigmoid(y_test_pred_logits).flatten()
        # 3. Wahrscheinlichkeiten --> Klassen [0 oder 1]
        y_test_pred_classes = (y_test_pred_probs >= THRESHOLD).float()
y_test_pred_classes

#%% Konfusionsmatrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(
    y_true=np.array(y_test_batch),
    y_pred=np.array(y_test_pred_classes))
# number format without decimal places
sns.heatmap(cm, annot=True, fmt="d")

#%% Genauigkeit ermitteln
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(
    y_true=np.array(y_test_batch),
    y_pred=np.array(y_test_pred_classes))
print(f"Accuracy: {accuracy}")

#%% Dummy-Classifier for comparison (without sklearn but Counter)
from collections import Counter
Counter(y_test)[0] / len(y_test)
