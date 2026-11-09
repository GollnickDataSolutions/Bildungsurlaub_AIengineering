#%% Pakete
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import kagglehub
import seaborn as sns
import pandas as pd
import os

#%% Download latest version
path = kagglehub.dataset_download("arunjangir245/boston-housing-dataset")

print("Path to dataset files:", path)
# %% path / BostonHousing.csv
full_path = os.path.join(path, "BostonHousing.csv")
df_boston = pd.read_csv(full_path)
df_boston

#%% Löschen von Zeilen mit nan
df_boston.dropna(inplace=True)
print(f"df boston shape:{df_boston.shape}")


#%% Behandlung von kategorischen Daten
df_dummies = pd.get_dummies(data=df_boston, columns=["rad"], dtype=float, drop_first=True)

#%% Exploratory Data Analysis
df_dummies.shape  # gibt Zeilen und Spalten zurück

df_dummies.describe()

#%% Correlation coefficients
# boston_corr = df_boston.corr()
# sns.heatmap(boston_corr, annot=True, annot_kws={"size":8})


#%% X...unabhängigen (beschreibende Merkmale), y...abhängige Variable (Zielgröße)
y = df_dummies['medv']
X = df_dummies.drop(columns=['medv'])
print(f"X shape: {X.shape}, y shape: {y.shape}")


#%% Data Sampling (Train, Test split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=42, shuffle=True)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


#%% Skalierung / Standardisierung
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
