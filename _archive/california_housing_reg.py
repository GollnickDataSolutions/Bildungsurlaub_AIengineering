#%% 
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
# %% data import
cal_housing = fetch_california_housing()
X = cal_housing["data"]
y = cal_housing["target"]

print(f"X shape: {X.shape}, y shape: {y.shape}")

#%% Train / Val / Test Split
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=2000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=2000, random_state=42)
print(f"{X_train.shape}, {y_train.shape}")

#%% Trainingsschleife nutzt train und val

#%% Evaluierung des Modells (auf Basis von test)