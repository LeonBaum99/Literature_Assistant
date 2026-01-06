from langchain_ollama import ChatOllama

from llmAG.llm_config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


def build_llm(model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
    return ChatOllama(model=model, temperature=temperature)
