#%% Pakete
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import kagglehub
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42)
X_train
# %% Standardisierung / Normalisierung
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()

X_train_scaled = np.array(scaler.fit_transform(X_train), dtype=np.float32)
X_test_scaled = np.array(scaler.transform(X_test), dtype=np.float32)

#%% Umwandlung der Daten in Tensoren
X_train_tensor = torch.from_numpy(X_train_scaled)
X_test_tensor = torch.from_numpy(X_test_scaled)
y_train_tensor = torch.from_numpy(np.array(y_train, dtype=np.float32))
y_test_tensor = torch.from_numpy(np.array(y_test, dtype=np.float32))

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
input_dim = X_train_scaled.shape[1]
output_dim = y_train.shape[1]
HIDDEN_DIM = 10
model = BostonRegressionModel(
    input_dim=input_dim, 
    output_dim=output_dim, 
    hidden_dim=HIDDEN_DIM)  

#%% TODO: Anzahl der Parameter ermitteln

#%% Optimierer
LR = 0.01
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

#%% Verlustfunktion
loss_fn = torch.nn.MSELoss()

# %% gesamte Trainingsschleife implementieren
# Schleife über die Anzahl der Epochen definieren
NUM_EPOCHS = 1000
losses_train, losses_test = [], []

for epoch in range(NUM_EPOCHS):
    # 1. Setze Gradienten auf Null
    optimizer.zero_grad()

    # 2. Forward-Pass (Berechnung der Vorhersagen)
    y_train_pred = model(X_train_tensor)

    # 3. Verluste ermitteln
    loss_train = loss_fn(y_train_pred, y_train_tensor)

    # 4. Gradienten berechnen
    loss_train.backward()

    # 5. Modellgewichte anpassen
    optimizer.step()

    # Trainings-Verluste wegspeichern (optional)
    losses_train.append(loss_train.item())

    # Test-Verluste ermitteln
    with torch.no_grad():
        y_test_pred = model(X_test_tensor)  # Vorhersage erstellen
        loss_test = loss_fn(y_test_pred, y_test_tensor)  # Verlust ermitteln
        losses_test.append(loss_test.item()) # Verlust abspeichern

# %% Trainings- und Testverluste visualisieren
sns.lineplot([losses_train, losses_test])

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

