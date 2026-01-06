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


class RagRequest(BaseModel):
    question: str
    n_results: int = 4
    model_name: str = "bert"
    include_sources: bool = True
    llm_model: Optional[str] = None
    temperature: Optional[float] = None


class RagResponse(BaseModel):
    answer: str
    template: str
    status: str
    needs_search: bool
    sources: Optional[List[SearchResult]] = None


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

class RecommendationRequest(BaseModel):
    positive_paper_ids: List[str]
    negative_paper_ids: List[str] = []
    limit: int = 10

class AuthorInfo(BaseModel):
    authorId: Optional[str]
    name: str

class RecommendedPaper(BaseModel):
    paperId: str
    title: str
    year: Optional[int] = None
    url: Optional[str] = None
    authors: List[AuthorInfo] = []
    abstract: Optional[str] = None

class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedPaper]

class PaperSearchResponse(BaseModel):
    query: str
    paperId: Optional[str] = None
