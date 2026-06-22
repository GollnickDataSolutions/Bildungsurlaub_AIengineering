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
        self.linear = nn.Linear(in_features=input_size, out_features=output_size)

    def forward(self, x):
        x = self.linear(x)
        #...
        return x

# %% Modellinstanz erstellen
input_size = X_train_scaled_tensor.shape[1]
output_size = y_train_tensor.shape[1]
model = RegressionTorch(input_size=input_size, output_size=output_size)
# %% Verlustfunktion
loss_fun = nn.MSELoss()

#%% Optimierer festlegen
LR = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#%% Trainingsschleife
EPOCHS = 50
losses = []
for epoch in range(EPOCHS):
    # Gradienten nullen
    optimizer.zero_grad()

    # Forward Pass
    y_train_pred = model(X_train_scaled_tensor)

    # Verluste berechnen
    # Ensure both tensors are float32 and have the same shape for the loss calculation
    loss = loss_fun(y_train_pred, y_train_tensor)

    # Backward Pass
    loss.backward()

    # Parameter updaten
    optimizer.step()

    # Verluste speichern für spätere Visualisierung
    losses.append(loss)

    print(f"Epoche {epoch}, Verlust {loss}")
# %%
