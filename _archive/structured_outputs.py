#%% Pakete
from dotenv import load_dotenv
from pprint import pprint
import os
from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
load_dotenv()

#%% Ausgabe formatieren
class MovieOutput(BaseModel):
    title: str
    main_actors: list[str]
    director: str = Field(
        description="Name des Regisseurs des Films", 
        examples=["Spielberg, Steven", "Cameron, James"])
    release_year: int

class MoviesOutput(BaseModel):
    movies: list[MovieOutput]
#%% Output Parser definieren
parser = PydanticOutputParser(pydantic_object=MoviesOutput)
pprint(parser.get_format_instructions())
#%% Prompt Template
messages = [
    ("system", """
       Du bist ein Filmfan und lieferst strukturierte Informationen zu den drei kommerziell erfolgreichsten Filmen.
       Der Nutzer übergibt dir eine Kurzbeschreibung der Handlung.
       Halte dich strikt bei der Ausgabe an die Formatanweisungen: {format_instruktionen}
    """),
    ("user", "Handlung: {handlung}")
]

prompt_template = ChatPromptTemplate.from_messages(messages=messages).partial(format_instruktionen=parser.get_format_instructions())
prompt_template

#%% Modellauswahl und Modellinstanz erstellen
MODEL_NAME = "deepseek/deepseek-v4-flash"
model = ChatOpenRouter(model=MODEL_NAME)


#%% Chain erstellen
chain = prompt_template | model | parser

#%% Modellinferenz
res = chain.invoke({'handlung': 'ein Schiff sinkt'})

#%%
res.model_dump()

# %%
