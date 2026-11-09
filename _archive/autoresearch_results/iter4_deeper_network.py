#%% Pakete
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# %% Hyperparameter
EPOCHS = 150
BATCH_SIZE = 128
LR = 0.001
TEST_SIZE = 0.1
VAL_SIZE = 0.1

#%% Daten laden
df = pd.read_csv("../data/coffee_shop_revenue.csv")
X = df.drop(columns=["Daily_Revenue"])
y = df["Daily_Revenue"]

# %% data splitting
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=42)

#%% scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# %% Tensoren
X_train_scaled_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_scaled_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
X_test_scaled_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(np.array(y_train).reshape(-1, 1), dtype=torch.float32)
y_val_tensor = torch.tensor(np.array(y_val).reshape(-1, 1), dtype=torch.float32)
y_test_tensor = torch.tensor(np.array(y_test).reshape(-1, 1), dtype=torch.float32)

#%% Dataset Klasse
class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = RegressionDataset(X=X_train_scaled_tensor, y=y_train_tensor)
val_dataset = RegressionDataset(X=X_val_scaled_tensor, y=y_val_tensor)
test_dataset = RegressionDataset(X=X_test_scaled_tensor, y=y_test_tensor)

#%% Dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

# %% Modellklasse
class RegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionTorch, self).__init__()
        self.HIDDEN1 = 128
        self.HIDDEN2 = 64
        self.HIDDEN3 = 64
        self.HIDDEN4 = 32
        self.HIDDEN5 = 16
        self.linear_0 = nn.Linear(in_features=input_size, out_features=128)
        self.bn_0 = nn.BatchNorm1d(128)
        self.relu_0 = nn.ReLU()
        self.dropout_0 = nn.Dropout(p=0.1)
        self.linear_1 = nn.Linear(in_features=128, out_features=64)
        self.bn_1 = nn.BatchNorm1d(64)
        self.relu_1 = nn.ReLU()
        self.dropout_1 = nn.Dropout(p=0.1)
        self.linear_2 = nn.Linear(in_features=64, out_features=64)
        self.bn_2 = nn.BatchNorm1d(64)
        self.relu_2 = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=0.1)
        self.linear_3 = nn.Linear(in_features=64, out_features=32)
        self.bn_3 = nn.BatchNorm1d(32)
        self.relu_3 = nn.ReLU()
        self.dropout_3 = nn.Dropout(p=0.1)
        self.linear_4 = nn.Linear(in_features=32, out_features=16)
        self.bn_4 = nn.BatchNorm1d(16)
        self.relu_4 = nn.ReLU()
        self.dropout_4 = nn.Dropout(p=0.1)
        self.linear_out = nn.Linear(in_features=16, out_features=output_size)

    def forward(self, x):
        x = self.linear_0(x)
        x = self.bn_0(x)
        x = self.relu_0(x)
        x = self.dropout_0(x)
        x = self.linear_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        x = self.linear_2(x)
        x = self.bn_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        x = self.linear_3(x)
        x = self.bn_3(x)
        x = self.relu_3(x)
        x = self.dropout_3(x)
        x = self.linear_4(x)
        x = self.bn_4(x)
        x = self.relu_4(x)
        x = self.dropout_4(x)
        x = self.linear_out(x)
        return x

# %% Modellinstanz
input_size = train_dataset.X.shape[1]
output_size = train_dataset.y.shape[1]
model = RegressionTorch(input_size=input_size, output_size=output_size)
total_params = sum(p.numel() for p in model.parameters())
print(f"Iteration Config: {json.dumps({"EPOCHS": 150, "BATCH_SIZE": 128, "LR": 0.001, "HIDDEN1": 128, "HIDDEN2": 64, "HIDDEN3": 64, "HIDDEN4": 32, "HIDDEN5": 16, "OPTIMIZER": "Adam", "DROPOUT": 0.1, "USE_BATCHNORM": true, "LR_SCHEDULER": null, "WEIGHT_DECAY": 1e-05, "desc": "Add more layers + batch normalization for deeper feature extraction"})}")
print(f"Total parameters: {total_params}")

# %% Verlustfunktion & Optimierer
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-05)


#%% Trainingsschleife
losses_train, losses_val = [], []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for X_train_batch, y_train_batch in train_loader:
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_train_batch)
        loss = loss_fun(y_pred, y_train_batch)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    losses_train.append(loss_epoch / len(train_loader))

    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            y_pred = model(X_val_batch)
            loss_val = loss_fun(y_pred, y_val_batch)
            val_loss_epoch += loss_val.item()
    losses_val.append(val_loss_epoch / len(val_loader))


    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"Epoche {epoch:3d}, Train: {losses_train[-1]:.4f}, Val: {losses_val[-1]:.4f}")

# %% Test
model.eval()
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        y_test_pred = model(X_test_batch).numpy()
        y_test_np = y_test_batch.numpy()

from sklearn.metrics import r2_score
r2 = r2_score(y_pred=y_test_pred.flatten(), y_true=y_test_np)
print(f"\nR2-Score auf Testdaten: {r2:.4f}")

# Save results
import json
results = {
    "r2": round(float(r2), 4),
    "final_train_loss": round(float(losses_train[-1]), 2),
    "final_val_loss": round(float(losses_val[-1]), 2),
    "total_params": total_params,
    "config": {"EPOCHS": 150, "BATCH_SIZE": 128, "LR": 0.001, "HIDDEN1": 128, "HIDDEN2": 64, "HIDDEN3": 64, "HIDDEN4": 32, "HIDDEN5": 16, "OPTIMIZER": "Adam", "DROPOUT": 0.1, "USE_BATCHNORM": true, "LR_SCHEDULER": null, "WEIGHT_DECAY": 1e-05, "desc": "Add more layers + batch normalization for deeper feature extraction"}
}
with open("c:/Temp/Bildungsurlaub_AIengineering/unsere_skripte/autoresearch_results/iter4_deeper_network.py".replace(".py", "_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved.")
