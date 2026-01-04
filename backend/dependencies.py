from functools import lru_cache

import chromadb

from backend.config import settings
from backend.services.embedder import EmbeddingService
from backend.services.processor import PDFProcessorService
from backend.services.recommendation import SemanticScholarService
from backend.services.vector_db import VectorDBService
from embeddingModels.ModernBertEmbedder import ModernBertEmbedder
from embeddingModels.QwenEmbedder import QwenEmbedder
from pdfProcessing.doclingTest import setup_docling_converter


class GlobalState:
    def __init__(self):
        self.chroma_client = None
        self.converter = None
        self.embedders = {}
        self.collections = {}


state = GlobalState()


def get_chroma_client():
    if not state.chroma_client:
        # USE CONFIG HERE
        print(f"Connecting to ChromaDB at {settings.chroma.path}...")
        state.chroma_client = chromadb.PersistentClient(path=settings.chroma.path)
    return state.chroma_client


def get_converter():
    if not state.converter:
        print("Loading Docling Converter...")
        state.converter = setup_docling_converter()
    return state.converter


def get_embedder(model_name: str):
    if model_name not in state.embedders:
        print(f"Loading Embedding Model: {model_name}...")

        if model_name == "bert":
            state.embedders["bert"] = ModernBertEmbedder(
                model_name=settings.models.bert,
                normalize=True
            )
        elif model_name == "qwen":
            state.embedders["qwen"] = QwenEmbedder(
                model_name=settings.models.qwen,
                use_fp16=True
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    return state.embedders[model_name]


@lru_cache()
def get_db_service() -> VectorDBService:

    return VectorDBService(
        db_path=settings.chroma.path,
        collection_names=settings.chroma.collections
    )


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache()
def get_pdf_service() -> PDFProcessorService:
    return PDFProcessorService()

@lru_cache()
def get_recommendation_service() -> SemanticScholarService:
    return SemanticScholarService(api_key=settings.semantic_scholar_api_key)
