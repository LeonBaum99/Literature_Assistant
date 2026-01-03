from typing import List, Dict, Any, Optional

from pydantic import BaseModel


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


class DeleteRequest(BaseModel):
    model_name: str = "bert"
    doc_ids: List[str]


class ResetRequest(BaseModel):
    model_name: str = "bert"


class StatusResponse(BaseModel):
    status: str
    message: str
    count: Optional[int] = 0


class StatsResponse(BaseModel):
    model_name: str
    count: int


class InspectResponse(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any]


class IDListResponse(BaseModel):
    model_name: str
    ids: List[str]
    total_in_batch: int

class EmbedDebugRequest(BaseModel):
    text: str
    model_name: str = "bert"

class EmbedDebugResponse(BaseModel):
    model_name: str
    vector_preview: List[float]
    dimension: int