#%% Pakete
from dotenv import load_dotenv
import os
from langchain_openrouter import ChatOpenRouter
load_dotenv()

#%% Prüfen, ob der API Key verfügbar ist
os.getenv("OPENROUTER_API_KEY")

#%% Modellauswahl und Modellinstanz erstellen
MODEL_NAME = "deepseek/deepseek-v4-flash"
model = ChatOpenRouter(model=MODEL_NAME)

#%% Modellinferenz
prompt = "Was ist aktuelle Datum?"
res = model.invoke(prompt)

#%%
res.model_dump()
# %%
res.content
