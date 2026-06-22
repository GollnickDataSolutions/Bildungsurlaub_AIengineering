#%% Pakete
import torch
import torch.nn as nn
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %%
X, y = fetch_california_housing(as_frame=True, return_X_y=True)
X.shape

#%% Exploratorive Datenanalyse
# y.head()
X.describe()
# X.isna().any(axis=1).sum()

# %% data splitting
# -> Ziel ist die Dreiteilung der Daten (train, validation, test)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=1000, random_state=42)
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1000, random_state=42)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

#%% scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# %% Umwandlung Dataframe --> Numpy Array --> Tensor
X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_scaled_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_scaled_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# Problem y_val.shape (1000, ) muss aber (1000, 1) sein
# dafür verwenden wir .reshape(-1, 1)
y_train_tensor = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float32)
y_val_tensor = torch.tensor(np.array(y_val).reshape(-1, 1), dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test).reshape(-1, 1), dtype=torch.float32)

# %% Modellklasse erstellen
class RegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionTorch, self).__init__()
        self.HIDDEN1 = 50
        self.HIDDEN2 = 30
        self.HIDDEN3 = 30
        self.HIDDEN4 = 20  # added fourth hidden layer
        self.linear_in = nn.Linear(in_features=input_size, out_features=self.HIDDEN1)
        self.linear_hidden1 = nn.Linear(in_features=self.HIDDEN1, out_features=self.HIDDEN2)
        self.relu = nn.ReLU()
        self.linear_hidden2 = nn.Linear(in_features=self.HIDDEN2, out_features=self.HIDDEN3)
        self.linear_hidden3 = nn.Linear(in_features=self.HIDDEN3, out_features=self.HIDDEN4)
        self.linear_out = nn.Linear(in_features=self.HIDDEN4, out_features=output_size)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_hidden1(x)
        x = self.relu(x)
        x = self.linear_hidden2(x)
        x = self.relu(x)
        x = self.linear_hidden3(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x
   
   

# %% Modellinstanz erstellen
input_size = X_train_scaled_tensor.shape[1]
output_size = y_train_tensor.shape[1]
model = RegressionTorch(input_size=input_size, output_size=output_size)
for name, param in model.named_parameters():
    print(f"Parameter '{name}':", param.data)
    
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of model parameters: {total_params}")
# %% Verlustfunktion
loss_fun = nn.MSELoss()

#%% Optimierer festlegen
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#%% Trainingsschleife
EPOCHS = 300
BATCH_SIZE = 128
losses_train, losses_val = [], []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for i in range(0, X_train_scaled.shape[0], BATCH_SIZE):
        model.train()
        # Gradienten nullen
        optimizer.zero_grad()

        # Forward Pass
        y_train_pred = model(X_train_scaled_tensor[i:i+BATCH_SIZE])

        # Verluste berechnen
        # Ensure both tensors are float32 and have the same shape for the loss calculation
        loss = loss_fun(y_train_pred, y_train_tensor[i:i+BATCH_SIZE])
        loss_epoch += loss.item()

        # Backward Pass
        loss.backward()

        # Parameter updaten
        optimizer.step()

    # Verluste speichern für spätere Visualisierung
    losses_train.append(loss_epoch)

    print(f"Epoche {epoch}, Train-Verlust {loss}")

    # Validierung
    model.eval()
    with torch.no_grad():
        y_val_pred = model(X_val_scaled_tensor)
        loss_val = loss_fun(y_val_pred, y_val_tensor)
        losses_val.append(loss_val.item())
        print(f"Epoche {epoch}, Val-Verlust {loss_val}")

# %%
import seaborn as sns
sns.lineplot([losses_train, losses_val])
# %%
for name, param in model.named_parameters():
    print(f"Parameter '{name}':", param.data)
# %% Modell final testen mit Testdaten
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test_scaled_tensor).numpy()
    
    
# %% Regressionsplot für echte Werte und Vorhersagen
sns.regplot(x=y_test, y=y_test_pred.flatten())
# %%
from sklearn.metrics import r2_score
r2_score(y_pred=y_test_pred.flatten(), y_true=y_test)

# %% Parameterstudie
# Einfluss von Parametern testen
# - Learning Rate
# - Anzahl Epochen
# - Verlustfunktion
# - Optimierer
