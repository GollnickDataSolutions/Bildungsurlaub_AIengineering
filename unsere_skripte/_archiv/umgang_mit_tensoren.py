#%% Pakete
import torch
# %% Erstellung eines Tensors
x = torch.tensor(5.5)

#%% einfache Berechnungen exakt wie bei numpy
y = x + 10
y

#%%
x = torch.tensor(2.0, requires_grad=True)
x.requires_grad

#%%
y = (x-3 ) * (x-6) * (x-4)
y.backward()

#%%
x.grad