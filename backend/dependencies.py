from functools import lru_cache
from backend.services.vector_db import VectorDBService
from backend.services.embedder import EmbeddingService
from backend.services.processor import PDFProcessorService

# Configuration
# TODO: Move to env vars or config file
CHROMA_PATH = "./chroma_db"
COLLECTION_MAP = {"bert": "scientific_papers_bert", "qwen": "scientific_papers_qwen"}

# --- Dependency Providers ---

@lru_cache()
def get_db_service() -> VectorDBService:
    return VectorDBService(db_path=CHROMA_PATH, collection_names=COLLECTION_MAP)

@lru_cache()
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()

@lru_cache()
def get_pdf_service() -> PDFProcessorService:
    return PDFProcessorService()