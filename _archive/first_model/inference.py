#%% Pakete
import torch
from data_prep import X_train_scaled, X_test_scaled, y_train, y_test


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
    input_size=20, 
    output_size=1)
# %%
state_dict = torch.load("FirstModel.pt")
model.load_state_dict(state_dict)
# state_dict

#%% Modellinferenz durchführen
input_data = X_test_scaled[-1, :].reshape(1, -1)
input_data_tensor = torch.tensor(input_data).float()
input_data.shape
#%%
model(input_data_tensor)