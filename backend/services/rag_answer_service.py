from typing import List, Optional

from langchain_core.documents import Document

from backend.services.embedder import EmbeddingService
from backend.services.vector_db import VectorDBService
from llmAG.rag.pipeline import RagPipeline, RagResponse
from llmAG.rag.retrievers import BaseRetriever


class ChromaRagRetriever(BaseRetriever):
    """Thin adapter that lets RagPipeline query Chroma via existing services."""

    def __init__(self, embed_service: EmbeddingService, db_service: VectorDBService, model_name: str):
        self._embed_service = embed_service
        self._db_service = db_service
        self._model_name = model_name

    def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        # Encode the query using the same embedding service used by /query.
        query_vec_np = self._embed_service.encode([query], model_name=self._model_name)
        # Query Chroma for the top-k most similar chunks.
        results = self._db_service.query(
            model_key=self._model_name,
            query_embedding=query_vec_np[0].tolist(),
            n_results=k
        )

        ids = results.get("ids", [[]])[0] if results else []
        documents = results.get("documents", [[]])[0] if results else []
        metadatas = results.get("metadatas", [[]])[0] if results else []
        distances = results.get("distances", [[]])[0] if results else []

        docs: List[Document] = []
        for doc_id, text, meta, score in zip(ids, documents, metadatas, distances):
            meta_dict = dict(meta) if meta else {}
            meta_dict["score"] = score
            meta_dict["id"] = doc_id
            docs.append(Document(page_content=text, metadata=meta_dict))
        return docs


def run_rag_answer(
        question: str,
        model_name: str,
        n_results: int,
        include_sources: bool,
        llm_model: Optional[str],
        temperature: Optional[float],
        embed_service: EmbeddingService,
        db_service: VectorDBService,
) -> RagResponse:
    """Run the RAG pipeline using existing embedding and Chroma services."""
    retriever = ChromaRagRetriever(embed_service, db_service, model_name)

    pipeline_kwargs = {}
    if llm_model:
        pipeline_kwargs["model"] = llm_model
    if temperature is not None:
        pipeline_kwargs["temperature"] = temperature

    pipeline = RagPipeline(retriever, **pipeline_kwargs)
    return pipeline.run(question, k=n_results, include_sources=include_sources)
