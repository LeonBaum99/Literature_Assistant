import os
from langchain_ollama import ChatOllama

from llmAG.llm_config import DEFAULT_MODEL, DEFAULT_TEMPERATURE


def build_llm(model: str = DEFAULT_MODEL, temperature: float = DEFAULT_TEMPERATURE):
    # # original: return ChatOllama(model=model, temperature=temperature)
    # Use host.docker.internal when running in Docker to reach host services
    base_url = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
    return ChatOllama(model=model, temperature=temperature, base_url=base_url)
