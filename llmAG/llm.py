from langchain_ollama import ChatOllama


def build_llm(model: str = "mistral-nemo", temperature: float = 0.2):
    return ChatOllama(model=model, temperature=temperature)
