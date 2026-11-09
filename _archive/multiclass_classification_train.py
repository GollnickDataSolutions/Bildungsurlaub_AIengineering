#%% pakete
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
# %% Hyperparameter
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.001
VAL_TEST_SIZE=1000
#%% Daten importieren
file_path = os.path.join("..", "data", "Network_logs.csv")
network_logs = pd.read_csv(file_path)
network_logs
#%% 
len(network_logs["Source_IP"].unique())
#%% Datenbereinigung
network_logs.drop(columns=["Source_IP", "Destination_IP"], inplace=True)
#%% Port ist numerisch codiert, aber repräsentiert eigentlich ein kategorisches Merkmal
network_logs["Port"] = network_logs["Port"].astype("str")
network_logs

#%% Intrusion Spalte löschen (Zielgröße von binärer Klassifizierung)
network_logs.drop(columns=["Intrusion"], inplace=True)

#%% One-Hot-Encoding / Dummy-Encoding
network_dummies = pd.get_dummies(network_logs.drop(columns=["Scan_Type"]), dtype=int, drop_first=True)
network_dummies
# %% unabhängige und abhängige Variablen trennen (X, y)
X = network_dummies
y = pd.factorize(network_logs["Scan_Type"])[0].astype(float)
print(f"Dimensionen von X und y: {X.shape}, {y.shape}")

#%% train / test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=VAL_TEST_SIZE, random_state=42)
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VAL_TEST_SIZE, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# %% datenskalierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_test_scaled
# %% Umwandlung Dataframe --> Numpy Array --> Tensor
X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_scaled_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_scaled_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# Problem y_val.shape (1000, ) muss aber (1000, 1) sein
# dafür verwenden wir .reshape(-1, 1)
y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.long)
y_val_tensor   = torch.tensor(np.array(y_val),   dtype=torch.long)
y_test_tensor  = torch.tensor(np.array(y_test),  dtype=torch.long)
#%% Dataset
class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0] # len(X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
train_dataset = ClassificationDataset(X=X_train_scaled_tensor, y=y_train_tensor)
val_dataset = ClassificationDataset(X=X_val_scaled_tensor, y=y_val_tensor)
test_dataset = ClassificationDataset(X=X_test_scaled_tensor, y=y_test_tensor)
#%% Dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=VAL_TEST_SIZE, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=VAL_TEST_SIZE, shuffle=False)

# %% Modellklasse erstellen
class NetworkClassificationModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(NetworkClassificationModel, self).__init__()
        self.HIDDEN1 = 8
        self.HIDDEN2 = 4
        self.linear_in = nn.Linear(in_features=input_size, out_features=self.HIDDEN1)
        self.linear_hidden1 = nn.Linear(in_features=self.HIDDEN1, out_features=self.HIDDEN2)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(in_features=self.HIDDEN2, out_features=output_size)
    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_hidden1(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x
   

# %% Modellinstanz erstellen
input_size = train_dataset.X.shape[1]
output_size = 3
model = NetworkClassificationModel(input_size=input_size, output_size=output_size)
#%%
for name, param in model.named_parameters():
    print(f"Parameter '{name}':", param.data)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of model parameters: {total_params}")

#%% Optimierer und Verlustfunktion
loss_fn = torch.nn.CrossEntropyLoss()
#%% Optimierer festlegen
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
 
# %% Trainingsschleife
#%% Trainingsschleife
losses_train, losses_val = [], []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i, (X_train_batch, y_train_batch) in enumerate(train_loader):
        model.train()
        # Gradienten nullen
        optimizer.zero_grad()

        # Forward Pass
        y_train_batch_pred = model(X_train_batch)

        # Verluste berechnen
        # Ensure both tensors are float32 and have the same shape for the loss calculation
        loss = loss_fn(y_train_batch_pred, y_train_batch)
        loss_epoch += loss.item()

        # Backward Pass
        loss.backward()

        # Parameter updaten
        optimizer.step()

    # Verluste speichern für spätere Visualisierung
    losses_train.append(loss_epoch/len(train_loader))

    print(f"Epoche {epoch}, Train-Verlust {loss}")

    # Validierung
    model.eval()
    with torch.no_grad():
        for j, (X_val_batch, y_val_batch) in enumerate(val_loader):
            y_val_batch_pred = model(X_val_batch)
            loss_val = loss_fn(y_val_batch_pred, y_val_batch)
            losses_val.append(loss_val.item()/len(val_loader))
            print(f"Epoche {epoch}, Val-Verlust {loss_val}")

# %% Verluste über Epochen visualisieren
import seaborn as sns
sns.lineplot([losses_train, losses_val])

# %% Vorhersagen für die Testdaten erstellen
model.eval()
with torch.no_grad():
    for (X_test, y_test) in test_loader:
        # Logits (Rohwahrscheinlichkeiten)
        y_test_pred_logits = model(X_test)
        # (echte) Wahrscheinlichkeiten mittels Sigmoid --> danach Wertebereich = [0, 1]
        y_test_pred_probs = torch.argmax(y_test_pred_logits, dim=1).numpy().flatten()

        y_test = y_test.numpy().flatten()

#%% aus Wahrscheinlichkeiten machen wir Klassen 
y_test_pred_cls = np.round(y_test_pred_probs)
# %% Konfusionsmatrix erstellen
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true=y_test, y_pred=y_test_pred_cls)
sns.heatmap(cm, annot=True, fmt="d")

# %%
