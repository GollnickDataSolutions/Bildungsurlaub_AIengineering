#%% Pakete
from langchain_ollama import ChatOllama

#%% Modellauswahl und Modellinstanz erstellen
MODEL_NAME = "qwen3-coder:latest"
model = ChatOllama(model=MODEL_NAME, temperature=1)

#%% Modellinferenz
prompt = "Was ist deine Lieblingszahl?"
res = model.invoke(prompt)

#%%
res.model_dump()
# %%