#%% Pakete
import torch


# %% Abgespeicherte, trainierte Modellgewichte ins Modell laden
trained_model_parameters = torch.load("BostonHousingModelWeights.pt")

#%% get model parameters of input layer
INPUT_DIM = trained_model_parameters['lin1.weight'].shape[1]
OUTPUT_DIM = trained_model_parameters['lin3.weight'].shape[0]
HIDDEN_DIM = trained_model_parameters['lin2.weight'].shape[1]
#%% Modellklasse
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
# %% Modellinstanz
model = BostonRegressionModel(
    input_dim=INPUT_DIM, 
    output_dim=OUTPUT_DIM,
    hidden_dim=HIDDEN_DIM)
#%% Zufällige Modellgewichte werden zugewiesen
model.state_dict()

#%% Modellgewichte laden
model.load_state_dict(state_dict=trained_model_parameters)
