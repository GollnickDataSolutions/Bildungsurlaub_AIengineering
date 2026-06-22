#%% Pakete
from data_prep import X_train_scaled, X_test_scaled, y_train, y_test
# %% 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter


#%%
writer = SummaryWriter(log_dir="runs")
#%% Hyperparameter
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 500


#%% Daten in Tensoren umwandeln
# X_train_tensor = torch.from_numpy(X_train_scaled).float()
# X_test_tensor = torch.from_numpy(X_test_scaled).float()
# y_train_tensor = torch.from_numpy(np.array(y_train).reshape(-1, 1)).float()
# y_test_tensor = torch.from_numpy(np.array(y_test).reshape(-1, 1)).float()
# print(f"X_train shape {X_train_tensor.shape}")
# print(f"y_train shape {y_train_tensor.shape}")
# print(f"y_test shape {y_test_tensor.shape}")

class LinRegDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(np.array(y).reshape(-1, 1)).float()

    def __len__(self):
        return len(self.X) # self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = LinRegDataset(X_train_scaled, y_train)
test_dataset = LinRegDataset(X_test_scaled, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


#%% Modellklasse erstellen
class LinRegTorch(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(LinRegTorch, self).__init__()
        self.linear_in = torch.nn.Linear(in_features=input_size, out_features=50)
        self.linear_out = torch.nn.Linear(in_features=50, out_features=1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = self.relu(x)
        return x
        
# %% Instanz der Modellklasse erstellen
model = LinRegTorch(
    input_size=X_train_scaled.shape[1], 
    output_size=1)

#%% Add graph to tensorboard
dummy_input = next(iter(train_loader))[0]
writer.add_graph(model, dummy_input)

#%% Gesamtanzahl der Parameter im Modell
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

#%% Optimizer und Verlustfunktion definieren
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()
# %% Modell-Trainingsschleife
import time
losses_train, losses_test = [], []
for epoch in range(EPOCHS):
    time.sleep(1)
    loss_epoch_train, loss_epoch_test = 0, 0
    for batch, (X_train_batch, y_train_batch) in enumerate(train_loader):
        # forward pass  -> Vorhersagen erstellt
        y_train_pred = model(X_train_batch)

        # berechne Verluste
        loss = loss_fn(y_train_batch, y_train_pred)

        # backward pass (Gradienten berechnen)
        loss.backward()

        # Modellparameter updaten
        optimizer.step()

        # Gradienten resetten
        optimizer.zero_grad()

        # speichere die Trainingsverluste in der Liste
        loss_epoch_train += loss.item()
    losses_train.append(loss_epoch_train / len(train_loader))
    # zeige den Trainingsstand mit Epoche und aktuellem Verlust
    print(f"Epoche {epoch}: Verlust-Train {loss.item()}")
    writer.add_scalar("train_loss", loss_epoch_train, global_step=epoch)

    # # Verluste für die Testdaten ermitteln
    for batch, (X_test_batch, y_test_batch) in enumerate(test_loader):
        with torch.no_grad():
            y_test_pred = model(X_test_batch)
            loss_test = loss_fn(y_test_batch, y_test_pred)
            loss_epoch_test += loss_test.item()
    losses_test.append(loss_epoch_test / len(test_loader))
    print(f"Epoche {epoch}: Verlust-Test {loss_test.item()}")
    writer.add_scalar("test_loss", loss_epoch_test, global_step=epoch)

# %% Visualisierung der Verluste
sns.lineplot(data=[losses_train, losses_test])

# %% Vorhersagen für die Testdaten erstellen
y_test_pred, y_test_true = [], []
for (X_test_batch, y_test_batch) in test_loader:
    with torch.no_grad():
        y_test_pred_batch = model(X_test_batch).flatten().numpy().tolist()
        y_test_pred.extend(y_test_pred_batch)
        y_test_true.extend(y_test_batch.flatten().numpy().tolist())


# %%
from sklearn.metrics import r2_score
r2_score(y_pred=y_test_pred, y_true=y_test_true)

# %% Speichern des Model state dictionaries
state_dictionary = model.state_dict()
torch.save(obj=state_dictionary, f="FirstModel.pt")
# %%
