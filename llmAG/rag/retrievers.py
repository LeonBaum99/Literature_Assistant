from __future__ import annotations

"""Retriever interface for the RAG pipeline."""

from typing import List

from langchain_core.documents import Document


class BaseRetriever:
    """Contract for retrievers used by the RAG pipeline."""

    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        raise NotImplementedError
