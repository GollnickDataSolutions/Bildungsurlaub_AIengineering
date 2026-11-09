#%% Pakete
import torch
import seaborn as sns
import numpy as np
# %% Tensor (von Hand) erstellen
x = torch.tensor(3.1415)
y = x + 25
#%% Ist die Gradientenberechnung aktiviert?
x.requires_grad
# False (standard, wenn Tensor von Hand erstellt wird)

#%% Gradientenberechnung aktivieren
x.requires_grad_()
# %%
def y_function(val):
    return (val-3) * (val-6) * (val-4)

x_range = np.linspace(0, 10, 101)
x_range
y_range = [y_function(i) for i in x_range]
sns.lineplot(x=x_range, y=y_range)
# %%
y = (x-3) * (x-6) * (x-4)
print(y)

#%% Steigung am Punkt x berechnen
y.backward()
x.grad