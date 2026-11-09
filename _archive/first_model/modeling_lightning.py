#%% Pakete
from data_prep import X_train_scaled, X_test_scaled, y_train, y_test
# %%
import torch
from torch.utils.data import Dataset, DataLoader
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import numpy as np

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


train_ds = LinRegDataset(X=X_train_scaled, y=y_train)
test_ds = LinRegDataset(X=X_test_scaled, y=y_test)

# %% DataLoader
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, BATCH_SIZE)
# %% PyTorch Lightning Klasse
class LinRegLightning(pl.LightningModule):
    def __init__(self, input_size, output_size, learning_rate=LEARNING_RATE):
        super().__init__()
        self.linear_in = torch.nn.Linear(in_features=input_size, out_features=50)
        self.linear_out = torch.nn.Linear(in_features=50, out_features=output_size)
        self.relu = torch.nn.ReLU()
        self.loss_fn = torch.nn.MSELoss()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_out(x)
        x = self.relu(x)
        return x

    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        # forward pass
        y_pred = self(X_batch)
        # loss calc
        loss = self.loss_fn(y_pred, y_batch)
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss
    
    def val_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        # forward pass
        y_pred = self(X_batch)
        # loss calc
        loss = self.loss_fn(y_pred, y_batch)
        self.log("val_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    # optionale Methoden
    def test_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        # forward pass
        y_pred = self(X_batch)
        # loss calc
        loss = self.loss_fn(y_pred, y_batch)
        self.log("test_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        y_pred = self(X_batch)
        return {"y_pred": y_pred, "y_true": y_batch}

# %%
model = LinRegLightning(input_size=X_train_scaled.shape[1], output_size= 1)

#%% Early Stopping
early_stopping = EarlyStopping(monitor="train_loss", patience=20, mode="min", min_delta=0.0, verbose=True)

# %% Trainer und Start des Trainings
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    callbacks = [early_stopping],
    log_every_n_steps=10
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_loader)
# %% Metriken anschauen
trainer.callback_metrics

#%% Vorhersagen für Testdaten erstellen
test_results = trainer.predict(model=model, dataloaders=test_loader)
test_results

#%% Modellgewichte speichern
model_statedict = model.state_dict()
torch.save(model_statedict, f="model_lightning.pt")

#%% Lightning Checkpoint abspeichern
trainer.save_checkpoint("FirstModelLightning.ckpt")

#%% Modellgewichte vom Checkpoint laden
model = LinRegLightning.load_from_checkpoint("FirstModelLightning.ckpt")
# %%