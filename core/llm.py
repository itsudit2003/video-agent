import os
from langchain_ollama import ChatOllama


import os
from langchain_ollama import ChatOllama


def get_llm(temperature: float = 0.3, **kwargs):
    return ChatOllama(
        model="gemma4:e4b",
        base_url=os.environ["OLLAMA_BASE_URL"].rstrip("/"),   # ends in /ollama
        client_kwargs={"headers": {"x-key": os.environ["WHISPER_REMOTE_KEY"]}},
        temperature=temperature,
        num_ctx=8192,
        **kwargs,
    )