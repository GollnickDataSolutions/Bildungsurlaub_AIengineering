#%% Pakete
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import kagglehub
import time
from torch.utils.tensorboard import SummaryWriter

#%% SummaryWriter instanziieren
RUN_NAME = "run3"

writer = SummaryWriter(log_dir=f"runs/tensorboard/{RUN_NAME}")

# Trainingskonstanten
LR = 0.005
HIDDEN_DIM = 10
NUM_EPOCHS = 1000
BATCH_SIZE = 32
TEST_SIZE = 100
#%% Daten herunterladen

# Download latest version
path = kagglehub.dataset_download("arunjangir245/boston-housing-dataset")

print("Path to dataset files:", path)

df_boston = pd.read_csv(f"{path}/BostonHousing.csv")

#%% Spaltennamen
df_boston.columns.tolist()
#%% Statistik
df_boston.describe()
# %% Anzahl Zeilen und Spalten
len(df_boston) # Anzahlen Zeilen
len(df_boston.columns)  # Anzahl Spalten
df_boston.shape # Anzahl Zeilen / Spalten

#%% Datentypen
df_boston.info()

#%% Gibt es leere Zeilen / missing values?
df_boston.isna().sum()  # ja, es gibt 5 na in "rm"

#%% Zeilen mit leeren Werten löschen
df_boston.dropna(inplace=True)
print(df_boston.shape)


# %% Spalte 'rad' repräsentiert einen Index, der den Abstand zum nächsten Highway repräsentiert
df_boston['rad'].unique()
df_boston_dummies = pd.get_dummies(data=df_boston, columns=['rad'], dtype=int)
df_boston_dummies

#%% Correlation Coefficients
boston_corr = df_boston.corr()
sns.heatmap(boston_corr, annot=True, annot_kws={"size": 8})
# %% Trennung von unabhängigen Variablen (X) und abhängiger Variable (y)
X = df_boston_dummies.drop(columns=['medv'])
y = df_boston_dummies[['medv']]
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# %% Train / Test Aufteilung
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)
X_train
# %% Standardisierung / Normalisierung
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()

X_train_scaled = np.array(scaler.fit_transform(X_train), dtype=np.float32)
X_test_scaled = np.array(scaler.transform(X_test), dtype=np.float32)

#%% Umwandlung der Daten in Tensoren
# X_train_tensor = torch.from_numpy(X_train_scaled)
# X_test_tensor = torch.from_numpy(X_test_scaled)
# y_train_tensor = torch.from_numpy(np.array(y_train, dtype=np.float32))
# y_test_tensor = torch.from_numpy(np.array(y_test, dtype=np.float32))

#%% Dataset
from torch.utils.data import Dataset, DataLoader
class BostonDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(np.array(y, dtype=np.float32))

    def __len__(self):
        return self.X.shape[0]  # returns number of rows

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = BostonDataset(X=X_train_scaled, y=y_train)
test_dataset = BostonDataset(X=X_test_scaled, y=y_test)

#%% DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=TEST_SIZE, shuffle=False)


# %% (abstrakte) Modellklasse erstellen
# class BostonRegressionModel(torch.nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(BostonRegressionModel, self).__init__()
#         self.lin1 = torch.nn.Linear(input_dim, output_dim)

#     def forward(self, x):
#         x = self.lin1(x)
#         return x        

class BostonRegressionModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(BostonRegressionModel, self).__init__()
        self.lin1 = torch.nn.Linear(input_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, output_dim)
        self.relu = torch.nn.ReLU() 

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        return x        
#%% Konkrete Instanz der Klasse erstellen
# control the seed in model creation
input_dim = X_train_scaled.shape[1]
output_dim = y_train.shape[1]
model = BostonRegressionModel(
    input_dim=input_dim, 
    output_dim=output_dim, 
    hidden_dim=HIDDEN_DIM)  

# add the model to add_graph, we need a dummy input
dummy_input = torch.randn(1, input_dim)
writer.add_graph(model, dummy_input)

#%%
for param in model.parameters():
    print(param)
#%% TODO: Anzahl der trainierbaren Parameter ermitteln
sum(p.numel() for p in model.parameters() if p.requires_grad)

#%% Optimierer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#%% Verlustfunktion
loss_fn = torch.nn.MSELoss()

# %% gesamte Trainingsschleife implementieren
# Schleife über die Anzahl der Epochen definieren
losses_train, losses_test = [], []

for epoch in range(NUM_EPOCHS):
    loss_train_epoch = 0
    time.sleep(2)
    for (X_train_batch, y_train_batch) in train_loader:
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
        writer.add_scalar("train_loss", loss_train_epoch, global_step=epoch)
    losses_train.append(loss_train_epoch)
    print(f"Epoch {epoch}, Loss: {loss_train_epoch}")

    # Test-Verluste ermitteln
    with torch.no_grad():
        for (X_test_batch, y_test_batch) in test_loader:
            y_test_pred = model(X_test_batch)  # Vorhersage erstellen
            loss_test = loss_fn(y_test_pred, y_test_batch)  # Verlust ermitteln
            losses_test.append(loss_test.item() / len(test_loader)) # Verlust abspeichern
            writer.add_scalar("test_loss", loss_test.item() / len(test_loader), global_step=epoch)

writer.close()

# %% Trainings- und Testverluste visualisieren
# leave out the first 10 observations
# make y axis logarithmic
import matplotlib.pyplot as plt
sns.lineplot([losses_train[20:], losses_test[10:]])
# plt.yscale('log')
plt.show()


# %% Modell-Evaluierung
# y_test_pred = model(X_test_tensor)
y_test_pred_np = y_test_pred.detach().numpy()
from sklearn.metrics import r2_score
r2_score(y_true=y_test['medv'], y_pred=y_test_pred_np)

# %% Ergebnisse
# Baseline-Model (Lineare Regression): R2=0.717
# Hidden=20, ReLU, R2=0.795
# Hidden=10, ReLU, R2=0.871
# Hidden=40, ReLU, R2=0.876
# 2x Hidden=10, ReLU, R2=0.89954

#%% 
torch.save(model.state_dict(), 'BostonHousingModelWeights.pt')