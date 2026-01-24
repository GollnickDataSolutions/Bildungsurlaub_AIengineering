#%% pakete
from langchain_ollama.chat_models import ChatOllama
import base64

#%% Modellinstanz erstellen
MODEL_NAME = "gemma3:4b"
model = ChatOllama(
    model=MODEL_NAME, 
    temperature=0.3
    )

#%%
res = model.invoke("Was ist Ollama?")
# %%
res.model_dump()

#%% Answer with Streaming
for chunk in model.stream("Was ist Bildungsurlaub?"):
    print(chunk.content, end="", flush=True)


#%% Encode the image in base64
image_path = "corgi_0.jpg"
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")
base64_image
#%% Prepare the message payload with text and image
user_query = "Was ist auf dem Bild zu sehen?"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_query},
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                },
            },
        ],
    }
]

# Invoke Ollama (assuming compatibility with this message format)
response = model.invoke(messages)
from pprint import pprint
pprint(response)


# %%
