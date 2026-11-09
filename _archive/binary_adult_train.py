#%% packages
import torch
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, precision_recall_curve)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from ucimlrepo import fetch_ucirepo
from collections import Counter

#%% Constants
BATCH_SIZE_TRAIN = 256
BATCH_SIZE_VAL = 1000
BATCH_SIZE_TEST = 1000
LR = 0.001
NUM_EPOCHS = 100

#%% Download Adult dataset from UCI ML Repository
print("Loading Adult dataset...")
adult = fetch_ucirepo(id=2)
X = adult.data.features
y = adult.data.targets

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y['income'].value_counts()}")

#%% Convert column names to lowercase
X.columns = X.columns.str.lower()

#%% Drop uninformative columns
cols_to_delete = ["fnlwgt", "education-num", "native-country"]
X.drop(columns=cols_to_delete, inplace=True, errors='ignore')
print(f"Shape after dropping columns: {X.shape}")

#%% Convert target to binary (1 for >50K, 0 for <=50K)
y_binary = (y["income"] == ">50K").astype(int).values.flatten()

#%% Handle categorical data
X_dummies = pd.get_dummies(data=X, dtype=int, drop_first=True)
print(f"Shape after creating dummies: {X_dummies.shape}")

# Remove very rare categories
n_samples = X_dummies.shape[0]
X_dummies = X_dummies.loc[:, (X_dummies.sum() >= 5)]
print(f"Shape after feature selection: {X_dummies.shape}")

#%% Train / Validation / Test split
X_train, X_val_test, y_train, y_val_test = train_test_split(
    X_dummies, y_binary,
    test_size=BATCH_SIZE_VAL + BATCH_SIZE_TEST,
    random_state=42,
    stratify=y_binary)

X_test, X_val, y_test, y_val = train_test_split(
    X_val_test, y_val_test,
    test_size=BATCH_SIZE_TEST,
    random_state=42,
    stratify=y_val_test)

print(f"Train set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Train target distribution: {Counter(y_train)}")

#%% Scale data
scaler = StandardScaler()
X_train_scaled = np.array(scaler.fit_transform(X_train), dtype=np.float32)
X_val_scaled = np.array(scaler.transform(X_val), dtype=np.float32)
X_test_scaled = np.array(scaler.transform(X_test), dtype=np.float32)

#%% Dataset class
class AdultDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(np.array(y, dtype=np.float32))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = AdultDataset(X_train_scaled, y_train)
val_dataset = AdultDataset(X_val_scaled, y_val)
test_dataset = AdultDataset(X_test_scaled, y_test)

#%% Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TRAIN, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE_VAL, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=False)

#%% Model class with Dropout
class AdultClassificationModel(torch.nn.Module):
    def __init__(self, input_dim, hidden1, dropout_rate=0.2):
        super(AdultClassificationModel, self).__init__()
        self.layer_1 = torch.nn.Linear(input_dim, hidden1)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.layer_2 = torch.nn.Linear(hidden1, hidden1 // 2)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.output = torch.nn.Linear(hidden1 // 2, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.layer_1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.layer_2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.output(x)
        return x


INPUT_DIM = X_train_scaled.shape[1]
model = AdultClassificationModel(input_dim=INPUT_DIM, hidden1=64, dropout_rate=0.2)
print(f"Model input dimension: {INPUT_DIM}")

#%% Handle class imbalance with weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_binary), y=y_binary)
class_weights = torch.tensor(class_weights, dtype=torch.float32)
pos_weight = class_weights[1] / class_weights[0]
print(f"Class weights: {class_weights}")
print(f"Positive weight ratio: {pos_weight:.2f}")

#%% Loss function and optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

#%% Training loop with Early Stopping
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60)

losses_train, losses_val = [], []
patience = 15
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(NUM_EPOCHS):
    model.train()
    loss_train_epoch = 0

    for (X_train_batch, y_train_batch) in train_loader:
        optimizer.zero_grad()
        y_train_pred_batch = model(X_train_batch)
        loss_train_batch = loss_fn(y_train_pred_batch.flatten(), y_train_batch)
        loss_train_batch.backward()
        optimizer.step()
        loss_train_epoch += loss_train_batch.item() / len(train_loader)

    losses_train.append(loss_train_epoch)

    # Validation
    model.eval()
    val_loss_epoch = 0
    with torch.no_grad():
        for (X_val_batch, y_val_batch) in val_loader:
            y_val_pred = model(X_val_batch)
            loss_val = loss_fn(y_val_pred.flatten(), y_val_batch)
            val_loss_epoch += loss_val.item() / len(val_loader)

    losses_val.append(val_loss_epoch)
    print(f"Epoch {epoch:3d}, Train Loss: {loss_train_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}")

    # Early stopping
    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f"\nEarly stopping at epoch {epoch}")
        break

#%% Plot loss
plt.figure(figsize=(10, 6))
plt.plot(losses_train, label='Train Loss', linewidth=2)
plt.plot(losses_val, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training and Validation Loss', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

#%% Get predictions on test data
print("\n" + "="*60)
print("EVALUATING ON TEST DATA")
print("="*60)

model.eval()
y_test_pred_logits_all = []

with torch.no_grad():
    for (X_test_batch, y_test_batch) in test_loader:
        y_test_pred_logits = model(X_test_batch)
        y_test_pred_logits_all.append(y_test_pred_logits.detach())

y_test_pred_logits_all = torch.cat(y_test_pred_logits_all).flatten()
y_test_pred_probs = torch.sigmoid(y_test_pred_logits_all).numpy()

#%% Find optimal threshold
precisions, recalls, thresholds = precision_recall_curve(y_test, y_test_pred_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
best_threshold_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[best_threshold_idx] if best_threshold_idx < len(thresholds) else 0.5

print(f"Optimal threshold: {optimal_threshold:.3f}")

y_test_pred_classes = (y_test_pred_probs >= optimal_threshold).astype(int)

#%% Print class distribution
print("\nCLASS DISTRIBUTION:")
print(f"  Class 0 (<=50K): {Counter(y_test)[0]} ({Counter(y_test)[0]/len(y_test)*100:.1f}%)")
print(f"  Class 1 (>50K):  {Counter(y_test)[1]} ({Counter(y_test)[1]/len(y_test)*100:.1f}%)")

#%% Calculate metrics
baseline_accuracy = Counter(y_test)[0] / len(y_test)
accuracy = accuracy_score(y_test, y_test_pred_classes)
precision = precision_score(y_test, y_test_pred_classes, zero_division=0)
recall = recall_score(y_test, y_test_pred_classes, zero_division=0)
f1 = f1_score(y_test, y_test_pred_classes, zero_division=0)
roc_auc = roc_auc_score(y_test, y_test_pred_probs)

print("\n" + "="*60)
print("EVALUATION METRICS")
print("="*60)
print(f"Baseline (always predict 0): {baseline_accuracy:.4f}")
print(f"Accuracy:                    {accuracy:.4f} (improvement: {(accuracy-baseline_accuracy):.4f})")
print(f"Precision (class 1):         {precision:.4f}")
print(f"Recall (class 1):            {recall:.4f}")
print(f"F1-Score:                    {f1:.4f}")
print(f"ROC-AUC:                     {roc_auc:.4f}")

#%% Confusion matrix
cm = confusion_matrix(y_test, y_test_pred_classes)
print("\nCONFUSION MATRIX:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title(f'Confusion Matrix (Threshold={optimal_threshold:.3f})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# %%
