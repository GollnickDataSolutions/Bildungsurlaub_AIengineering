#%% Pakete
from dotenv import load_dotenv
from pprint import pprint
import os
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

#%% Prompt Template
messages = [
    ("system", """
       Du bist ein begnadeter Comedian im Stile von {comedian}.
       Der Nutzer übergibt dir ein Thema für das du den Witz erzählst.
    """),
    ("user", "Thema: {thema}")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages)
prompt_template

#%% Modellauswahl und Modellinstanz erstellen
MODEL_NAME = "deepseek/deepseek-v4-flash"
model = ChatOpenRouter(model=MODEL_NAME)

#%% Output Parser definieren
parser = StrOutputParser()

#%% Chain erstellen
chain = prompt_template | model | parser


#%% Modellinferenz
res = chain.invoke({'comedian': 'Volker Pispers', 'thema': 'Rentenreform'})

#%% Stream output
for chunk in chain.stream({'comedian': 'Volker Pispers', 'thema': 'Rentenreform'}):
    print(chunk, end="", flush=True)

#%%
pprint(res, width=40)


