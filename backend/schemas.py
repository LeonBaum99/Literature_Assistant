from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class QueryRequest(BaseModel):
    query_text: str
    n_results: int = 5
    model_name: str = "bert"  # Optional: allow switching models per query if needed

class SearchResult(BaseModel):
    doc_id: str
    score: float
    content: str
    metadata: Dict[str, Any]

class QueryResponse(BaseModel):
    results: List[SearchResult]

class IngestResponse(BaseModel):
    filename: str
    message: str
    chunks_added: int
    parent_id: str