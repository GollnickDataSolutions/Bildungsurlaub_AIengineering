#%% Pakete
from data_prep import X_train_scaled, X_test_scaled, y_train, y_test
# %%
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

#%% Hyperparameter
BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCHS = 500


#%% Dataset
class LinRegDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(np.array(y).reshape(-1, 1)).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = LinRegDataset(X_train_scaled, y_train)
test_dataset = LinRegDataset(X_test_scaled, y_test)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)


#%% LightningModule erstellen
class LinRegLightning(pl.LightningModule):
    def __init__(self, input_size, output_size, learning_rate=LEARNING_RATE):
        super().__init__()
        self.save_hyperparameters()
        self.linear_in = torch.nn.Linear(in_features=input_size, out_features=50)
        self.linear_out = torch.nn.Linear(in_features=50, out_features=output_size)
        self.relu = torch.nn.ReLU()
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = self.relu(x)
        return x

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        loss = self.loss_fn(y_pred, y_batch)
        self.log("test_loss", loss, on_epoch=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        return {"y_pred": y_pred, "y_true": y_batch}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


# %% Instanz der Modellklasse erstellen
model = LinRegLightning(
    input_size=X_train_scaled.shape[1],
    output_size=1,
)

#%% Gesamtanzahl der Parameter im Modell
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in the model: {total_params}")

#%% Early Stopping Callback
early_stop_callback = EarlyStopping(
    monitor="val_loss",
    patience=20,
    mode="min",
    min_delta=0.0,
    verbose=True,
)

#%% Trainer und Training
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks=[early_stop_callback],
    log_every_n_steps=10,
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)

# %% Letzte Metriken aus dem Training auslesen
metrics = trainer.callback_metrics
print(f"Letzte Metriken: {metrics}")

# %% Test-Loss über trainer.test() loggen
test_results = trainer.test(model=model, dataloaders=test_loader)
print(f"Test-Ergebnisse: {test_results}")

# %% Vorhersagen über trainer.predict() erstellen
predictions = trainer.predict(model=model, dataloaders=test_loader)

# Lightning gibt eine Liste pro Batch zurück — zu einem Tensor zusammenfügen
y_test_pred = torch.cat([p["y_pred"] for p in predictions]).flatten().numpy()
y_test_true = torch.cat([p["y_true"] for p in predictions]).flatten().numpy()

# %% R2 Score
from sklearn.metrics import r2_score
print(f"R2 Score: {r2_score(y_pred=y_test_pred, y_true=y_test_true)}")

# %% Speichern des Model state dictionaries
state_dictionary = model.state_dict()
torch.save(obj=state_dictionary, f="FirstModel.pt")

# %% Optional: Speichern des kompletten Lightning-Checkpoints
trainer.save_checkpoint("FirstModel.ckpt")
