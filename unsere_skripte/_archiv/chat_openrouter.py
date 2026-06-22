#%% paket
from dotenv import load_dotenv
import os
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent
from langchain_tavily import TavilySearch
load_dotenv()
# %%
MODEL = "deepseek/deepseek-v4-flash"
model = ChatOpenRouter(model=MODEL)

#%% Modell nutzen
# query = "Was ist Deepseek V4?"
query = "Wieviele e sind im Wort erdbeere?"
res = model.invoke(query)
# %%
res.model_dump()

#%%
from pprint import pprint
pprint(res.content)
# %%
search_tool = TavilySearch(max_results=5)
# search_tool.invoke(query)
# %%
agent = create_agent(
    model=model,
    tools=[search_tool],
    system_prompt="""
        Du sprichst wie ein Pirat.
    """
)
# %%
messages = [
    ("user", query)
]
res = agent.invoke({
    "messages": messages
})
# %%
res
# %%
res['messages'][-1].content
# %%
