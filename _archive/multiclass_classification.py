#%% Pakete
import kagglehub
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, f1_score, classification_report
import numpy as np

#%% Download latest version
path = kagglehub.dataset_download("developerghost/intrusion-detection-logs-normal-bot-scan")

print("Path to dataset files:", path)
file_path = os.path.join(path, "Network_logs.csv")
df = pd.read_csv(file_path)
df

#%% Data Analysis
df["Destination_IP"].value_counts()

#%% unabhängige und abhängige Features trennen
X = df[['Port', 'Request_Type', 'Protocol',
       'Payload_Size', 'User_Agent', 'Status']]
y = pd.factorize(df["Scan_Type"])[0].astype(float)
# X['Port'] = X['Port'].astype("str")

#%% Behandlung kategorischer Daten
X_dummy = pd.get_dummies(X, dtype=int, drop_first=True, columns=['Port', 'Request_Type', 'Protocol', 'User_Agent', 'Status'])

#%% Hyperparameter
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 64
# %%
X_trainval, X_test, y_trainval, y_test = train_test_split(X_dummy, y, test_size=2000, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=2000, random_state=42, stratify=y_trainval)

#%%
# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()
# scaler.fit_transform()


# %% Dataset und Dataloader
train_dataset = TensorDataset(torch.Tensor(X_train.values), torch.LongTensor(y_train))
val_dataset = TensorDataset(torch.Tensor(X_val.values), torch.LongTensor(y_val))
test_dataset = TensorDataset(torch.Tensor(X_test.values), torch.LongTensor(y_test))

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=2000, shuffle=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=2000, shuffle=False)

# %% Modellklasse
class MultiClassClassificationModel(torch.nn.Module):
    def __init__(self, input_size, output_size, hidden_size, dropout_rate=0.2):
        super(MultiClassClassificationModel, self).__init__()
        self.lin1 = torch.nn.Linear(input_size, hidden_size)
        self.lin2 = torch.nn.Linear(hidden_size, output_size)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        x = self.dropout(x)  # dropout for regularization
        x = self.lin2(x)
        return x

model = MultiClassClassificationModel(input_size=X_train.shape[1], output_size=3, hidden_size=HIDDEN_SIZE)

#%% Optimierer und Verlustfunktion
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()
# %%
train_losses = []
val_losses = []
for epoch in range(EPOCHS):
    train_loss = 0
    val_loss = 0

    # Training loop
    model.train()
    for (X_train_batch, y_train_batch) in train_loader:
        # forward pass
        y_train_pred = model(X_train_batch)

        # loss calc
        loss_train = loss_fn(y_train_pred, y_train_batch)

        # backward pass
        loss_train.backward()

        # weight update
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()
        
        # store losses
        train_loss += loss_train.item()

    train_losses.append(train_loss / len(train_loader))

    # Validation loop
    model.eval()
    with torch.no_grad():
        for (X_val_batch, y_val_batch) in val_loader:
            y_val_pred = model(X_val_batch)
            loss_val = loss_fn(y_val_pred, y_val_batch)
            val_loss += loss_val.item()
    val_losses.append(val_loss / len(val_loader))

    print(f"Epoch {epoch}: Loss-Train: {train_loss}, Loss-Val: {val_loss}")

# %% Verluste visualisieren
loss_df = pd.DataFrame({"train_loss": train_losses, "val_loss": val_losses})
sns.lineplot(data=loss_df)

# %% Vorhersagen für die Testdaten erstellen
y_test_true = []
y_test_pred_classes = []
y_test_pred_probs = []

model.eval()
with torch.no_grad():
    for X_test_batch, y_test_batch in test_loader:
        # Get logits from model
        logits = model(X_test_batch)
        # Apply softmax to get class probabilities
        probs = torch.softmax(logits, dim=1)
        # Predicted class = argmax over class dimension
        preds = torch.argmax(probs, dim=1)

        y_test_true.extend(y_test_batch.tolist())
        y_test_pred_classes.extend(preds.tolist())
        y_test_pred_probs.append(probs.numpy())

y_test_pred_probs = np.concatenate(y_test_pred_probs, axis=0)

# %% Confusion Matrix
cm = confusion_matrix(y_true=y_test_true, y_pred=y_test_pred_classes)
cm
# %%
sns.heatmap(cm, annot=True, fmt="d")

#%% Genauigkeit
accuracy_score(y_true=y_test_true, y_pred=y_test_pred_classes)

# %% Klassenverteilung im Testset
pd.Series(y_test_true).value_counts()

# %% Area under Curve (multiclass, one-vs-rest)
auc_value = roc_auc_score(y_true=y_test_true, y_score=y_test_pred_probs, multi_class="ovr")
print(f"AUC: {auc_value:.4f}")

#%% F1 Score (multiclass)
f1_score(y_true=y_test_true, y_pred=y_test_pred_classes, average="macro")

#%% Classification Report
print(classification_report(y_true=y_test_true, y_pred=y_test_pred_classes))
# %% ONNX model export
dummy_input = torch.randn((1, 26))
onnx_program = torch.onnx.export(model, dummy_input)

# %%
onnx_program.save("model.onnx")