

#%% Packages
# data preparation
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# modeling
import torch
from torch.utils.data import Dataset, DataLoader

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

#%% Hyper Parameter
BATCH_SIZE = 4
seq_len = 10

# %% Data Import
flights = sns.load_dataset("flights")
print(f'Number of Entries: {len(flights)}')
flights.head(2)

# %% Visualization
sns.lineplot(x=range(len(flights)), y='passengers', data=flights)
plt.title('Number of Flight Passengers per Month')
plt.xlabel('Month')
plt.ylabel('Number of Passengers')
plt.show()

# %% Convert year and month to datetime
flights["year_month"] = pd.to_datetime(flights["year"].astype(str) + "-" + flights["month"].astype(str))

#%% Scale the data
scaler = MinMaxScaler()
Xy = flights.passengers.values.astype(np.float32)
Xy_scaled = scaler.fit_transform(Xy.reshape(-1, 1))

# %% Data Restructuring
X_restruct = np.array([Xy_scaled[i:i+seq_len] for i in range(len(Xy_scaled) - seq_len)])
y_restruct = np.array([Xy_scaled[i+seq_len] for i in range(len(Xy_scaled) - seq_len)])
print(f'X_restruct shape: {X_restruct.shape}')
print(f'y_restruct shape: {y_restruct.shape}')
#%% train/test split
last_n_months = 12
clip_point = len(X_restruct) - last_n_months
X_train = X_restruct[:clip_point]
X_test = X_restruct[clip_point:]
y_train = y_restruct[:clip_point]
y_test = y_restruct[clip_point:]

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")
#%% Visualisierung des Train-/Test-Splits
plt.figure(figsize=(10, 6))
sns.lineplot(data=flights, x='year_month', y='passengers', label='Total Data')
# Hier musst du die Indizes an die Zeitstempel anpassen
split_date = flights['year_month'].iloc[clip_point + seq_len]
plt.axvline(x=split_date, color='red', linestyle='--', label='Train/Test Split')
plt.title('Number of Flight Passengers per Month')
plt.legend()
plt.show()
# %% Dataset
class FlightDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].squeeze(-1), self.y[idx]
#%% Dataloader
train_loader = DataLoader(FlightDataset(X_train, y_train), batch_size=BATCH_SIZE)
test_loader = DataLoader(FlightDataset(X_test, y_test), batch_size=len(y_test))



#%% Packages
# modeling
import numpy as np
import torch
from torch import nn

# visualization
import matplotlib.pyplot as plt
import seaborn as sns


#%% Hyper Parameter
EPOCHS = 100
BATCH_SIZE = 4
SEQ_LEN = 10
OUT_CHANNELS = 16
KERNEL_SIZE = 3
HIDDEN_SIZE = 50

#%% Model
class FlightModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, seq_len, hidden_size, output_size):
        super(FlightModel, self).__init__()
        # 1D Convolutional Layer zur Erkennung lokaler Muster
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        
        # Berechnung der Dimension nach der Faltung, um die lineare Schicht anzupassen
        conv_output_size = out_channels * (seq_len - kernel_size + 1)
        
        self.fc1 = nn.Linear(conv_output_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.relu(x)
        
        x = torch.flatten(x, start_dim=1)
        
        # Lineare Schichten anwenden
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        
        return x

# %% Model, Loss and Optimizer
model = FlightModel(
    in_channels=1,
    out_channels=OUT_CHANNELS,
    kernel_size=KERNEL_SIZE,
    seq_len=SEQ_LEN,
    hidden_size=HIDDEN_SIZE,
    output_size=1
)

loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

#%% Train
loss_train = []
for epoch in range(EPOCHS):
    loss_epoch = 0
    for j, train_batch in enumerate(train_loader):
        X_batch, y_batch = train_batch
       
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = loss_fun(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.item()
    loss_train.append(loss_epoch/len(train_loader))
    
    print(f"Epoch: {epoch}, Loss: {loss.data}")


#%% Loss Plot
sns.lineplot(x=range(EPOCHS), y=loss_train)
plt.xlabel('Epoche [-]')
plt.ylabel('Verlust [-]')
plt.title('Verlust')

# %% Create Predictions
X_test_torch, y_test_torch = next(iter(test_loader))
with torch.no_grad():
    y_pred = model(X_test_torch)
y_act = y_test_torch.numpy().squeeze()
x_act = range(y_act.shape[0])
sns.lineplot(x=x_act, y=y_act, label = 'Actual',color='black')
sns.lineplot(x=x_act, y=y_pred.squeeze(), label = 'Predicted',color='red')
sns.lineplot(x=x_act, y=y_act, label = 'Actual',color='black')
sns.lineplot(x=x_act, y=y_pred.squeeze(), label = 'Predicted',color='red')

#%% calculate rmse error
rmse = np.sqrt(np.mean((y_act - y_pred.squeeze().numpy())**2))
print(f"RMSE: {rmse:.2f}")

#%% calculate mape error
mape = np.mean(np.abs((y_act - y_pred.squeeze().numpy()) / y_act)) * 100
print(f"MAPE: {mape:.2f}%")

# %% correlation plot
sns.scatterplot(x=y_act, y=y_pred.squeeze(), label = 'Predicted',color='red', alpha=0.5)
# Add diagonal line with slope 1 and same range as data
plt.plot([0.5, 1.], [0.5, 1.], 'k--', alpha=0.5, label='Perfect Prediction')
plt.legend()
plt.title('Vorhersage vs. tatsächlicher Wert')
plt.xlabel('Tatsächlicher Wert')
plt.ylabel('Vorhergesagter Wert')




# %%