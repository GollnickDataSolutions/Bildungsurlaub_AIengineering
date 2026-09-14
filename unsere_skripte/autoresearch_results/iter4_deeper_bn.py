import torch, torch.nn as nn, numpy as np, pandas as pd, json
import json as _json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

EPOCHS = 150
BATCH_SIZE = 128
LR = 0.001

df = pd.read_csv("../data/coffee_shop_revenue.csv")
X = df.drop(columns=["Daily_Revenue"])
y = df["Daily_Revenue"]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1/0.9, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)
y_train_t = torch.tensor(np.array(y_train).reshape(-1,1), dtype=torch.float32)
y_val_t = torch.tensor(np.array(y_val).reshape(-1,1), dtype=torch.float32)
y_test_t = torch.tensor(np.array(y_test).reshape(-1,1), dtype=torch.float32)

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_ds = RegressionDataset(X_train_t, y_train_t)
val_ds = RegressionDataset(X_val_t, y_val_t)
test_ds = RegressionDataset(X_test_t, y_test_t)
train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, len(val_ds), shuffle=False)
test_loader = DataLoader(test_ds, len(test_ds), shuffle=False)

class RegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
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

input_size = train_ds.X.shape[1]
output_size = train_ds.y.shape[1]
model = RegressionTorch(input_size, output_size)
total_params = sum(p.numel() for p in model.parameters())
config_dict = {'EPOCHS': 150, 'BATCH_SIZE': 128, 'LR': 0.001, 'HIDDEN1': 128, 'HIDDEN2': 64, 'HIDDEN3': 64, 'HIDDEN4': 32, 'HIDDEN5': 16, 'OPTIMIZER': 'Adam', 'DROPOUT': 0.1, 'USE_BATCHNORM': True, 'LR_SCHEDULER': None, 'WEIGHT_DECAY': 1e-05}
print("Config:", _json.dumps(config_dict))
print(f"Params: {total_params}")

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-05)


losses_train, losses_val = [], []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for Xb, yb in train_loader:
        model.train()
        optimizer.zero_grad()
        yp = model(Xb)
        loss = loss_fn(yp, yb)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    losses_train.append(loss_epoch / len(train_loader))
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for Xb, yb in val_loader:
            yp = model(Xb)
            val_loss += loss_fn(yp, yb).item()
    losses_val.append(val_loss / len(val_loader))
    if epoch % 20 == 0 or epoch == EPOCHS - 1:
        print(f"E {epoch:3d} train: {losses_train[-1]:.2f} val: {losses_val[-1]:.2f}")

model.eval()
with torch.no_grad():
    for Xb, yb in test_loader:
        yp = model(Xb).numpy()
        yt = yb.numpy()

from sklearn.metrics import r2_score
r2 = r2_score(y_pred=yp.flatten(), y_true=yt)
print(f"\nR2-Score: {r2:.4f}")

res_path = r"c:\Temp\Bildungsurlaub_AIengineering\unsere_skripte\autoresearch_results\iter4_deeper_bn_results.json"
res = {"r2": round(float(r2), 4), "total_params": total_params}
with open(res_path, "w") as f:
    json.dump(res, f, indent=2)