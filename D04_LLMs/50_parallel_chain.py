#%% packages
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, graph_mermaid
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

#%% Model Instance
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=1)

#%% Prepare Prompts
# example: style variations (friendly, polite) vs. (savage, angry)
polite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Reply in a friendly and polite manner. answer in one sentence."),
    ("human", "{topic}")
])

savage_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Reply in a savage and angry manner. answer in one sentence."),
    ("human", "{topic}")
])

#%% Prepare Chains
polite_chain = polite_prompt | llm | StrOutputParser()
savage_chain = savage_prompt | llm | StrOutputParser()


# %% Runnable Parallel
map_chain = RunnableParallel(
    polite=polite_chain,
    savage=savage_chain
)

# %% Invoke
topic = "Was ist der Sinn des Lebens?"
result = map_chain.invoke({"topic": topic})
# %% Print
from pprint import pprint
pprint(result)
# %%


