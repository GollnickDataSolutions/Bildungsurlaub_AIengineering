#%% Pakete
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import kagglehub

# Trainingskonstanten
LR_RANGE = [0.001, 0.01, 0.1]
NUM_EPOCHS_RANGE = [100, 200, 300]
BATCH_SIZE_RANGE = [ 64, 128, 256]
HIDDEN_DIM = 10
TEST_SIZE = 100

#%% create df for all combinations of hyperparameters
import itertools
hyperparameter_combinations = list(itertools.product(LR_RANGE, NUM_EPOCHS_RANGE, BATCH_SIZE_RANGE))
df_hyperparameters = pd.DataFrame(hyperparameter_combinations, columns=['lr', 'num_epochs', 'batch_size'])
df_hyperparameters
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



def train(LR, NUM_EPOCHS, BATCH_SIZE, HIDDEN_DIM):
    # DataLoader

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=TEST_SIZE, shuffle=False)
    model = BostonRegressionModel(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=HIDDEN_DIM)  
    # Optimierer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # Verlustfunktion
    loss_fn = torch.nn.MSELoss()

    #  gesamte Trainingsschleife implementieren
    # Schleife über die Anzahl der Epochen definieren
    NUM_OBS = X_train_scaled.shape[0]
    losses_train, losses_test = [], []

    for epoch in range(NUM_EPOCHS):
        loss_train_epoch = 0
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
        losses_train.append(loss_train_epoch)
        # print(f"Epoch {epoch}, Loss: {loss_train_epoch}")

        # Test-Verluste ermitteln
        with torch.no_grad():
            for (X_test_batch, y_test_batch) in test_loader:
                y_test_pred = model(X_test_batch)  # Vorhersage erstellen
                loss_test = loss_fn(y_test_pred, y_test_batch)  # Verlust ermitteln
                losses_test.append(loss_test.item() / len(test_loader)) # Verlust abspeichern


    #  Modell-Evaluierung
    # y_test_pred = model(X_test_tensor)
    y_test_pred_np = y_test_pred.detach().numpy()
    from sklearn.metrics import r2_score
    r2 = r2_score(y_test.to_numpy(dtype=np.float32), y_test_pred_np)
    return r2

# %%
for index, row in df_hyperparameters.iterrows():
    LR = row['lr']
    NUM_EPOCHS = int(row['num_epochs'])
    BATCH_SIZE = int(row['batch_size'])
    r2 = train(LR, NUM_EPOCHS, BATCH_SIZE, HIDDEN_DIM)
    print(f"Hyperparameters set {index}: R2 Score: {r2}")
    df_hyperparameters.loc[index, 'R2'] = r2
# %% visualise the result as barplot
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(df_hyperparameters.index, df_hyperparameters['R2'], color='skyblue')
plt.title('R2 Score for Different Hyperparameter Sets')
plt.xlabel('Hyperparameter Set Index')
plt.ylabel('R2 Score')
plt.grid(True)
plt.show()

# %%
df_hyperparameters.sort_values(by='R2', ascending=False, inplace=True)
df_hyperparameters

#%% show a heatmap with x=lr, y=num_epochs, fill=R2, use only seaborn sns.heatmap
