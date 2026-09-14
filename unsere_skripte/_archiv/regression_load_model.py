#%% Pakete
import torch
import torch.nn as nn

#%% Modellklasse
class RegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(RegressionTorch, self).__init__()
        self.HIDDEN1 = 50
        self.HIDDEN2 = 30
        self.HIDDEN3 = 30
        self.HIDDEN4 = 20  # added fourth hidden layer
        self.linear_in = nn.Linear(in_features=input_size, out_features=self.HIDDEN1)
        self.linear_hidden1 = nn.Linear(in_features=self.HIDDEN1, out_features=self.HIDDEN2)
        self.relu = nn.ReLU()
        self.linear_hidden2 = nn.Linear(in_features=self.HIDDEN2, out_features=self.HIDDEN3)
        self.linear_hidden3 = nn.Linear(in_features=self.HIDDEN3, out_features=self.HIDDEN4)
        self.linear_out = nn.Linear(in_features=self.HIDDEN4, out_features=output_size)

    def forward(self, x):
        x = self.linear_in(x)
        x = self.relu(x)
        x = self.linear_hidden1(x)
        x = self.relu(x)
        x = self.linear_hidden2(x)
        x = self.relu(x)
        x = self.linear_hidden3(x)
        x = self.relu(x)
        x = self.linear_out(x)
        return x
# %% Modellinstanz erstellen
model = RegressionTorch(input_size=8, output_size=1)
model.state_dict()  # zufällige Modellgewichte

#%% trainierte Gewichte ins Modell laden
state_dict = torch.load("RegressionModel.pt")
state_dict
#%%
model.load_state_dict(state_dict=state_dict)

# %%
