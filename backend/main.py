import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException

from backend.dependencies import get_db_service, get_embedding_service, get_pdf_service, get_recommendation_service
# Import Internal Modules
from backend.schemas import (
    QueryRequest, QueryResponse, IngestResponse, SearchResult,
    DeleteRequest, ResetRequest, StatusResponse,
    StatsResponse, InspectResponse, IDListResponse,
    EmbedDebugRequest, EmbedDebugResponse, RecommendationRequest, RecommendationResponse, PaperSearchResponse,
    RagRequest, RagResponse
)
from backend.services.embedder import EmbeddingService
from backend.services.processor import PDFProcessorService
from backend.services.recommendation import SemanticScholarService
from backend.services.rag_answer_service import run_rag_answer
from backend.services.vector_db import VectorDBService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load services to ensure models are ready on startup
    get_db_service()
    get_embedding_service().load_model("bert")  # Pre-load default
    get_pdf_service()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(
        file: UploadFile = File(...),
        model_name: str = Form("bert"),
        # Dependency Injection
        pdf_service: PDFProcessorService = Depends(get_pdf_service),
        embed_service: EmbeddingService = Depends(get_embedding_service),
        db_service: VectorDBService = Depends(get_db_service),
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    # 1. Handle File (IO)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 2. Process PDF (Domain: Processor)
        metadata, sections = pdf_service.process_pdf(tmp_path)

        # 3. Prepare Data
        # Use 'or' to handle None values properly
        parent_id = metadata.get("arxiv_id") or file.filename or "unknown_doc"
        parent_id = parent_id.replace(" ", "_").replace(":", "_").replace("-", "_")
        docs, metas, ids = [], [], []

        base_meta = {
            "parent_id": parent_id,
            "filename": file.filename,
            "title": metadata.get("title", "Unknown"),
            "authors": ", ".join(metadata.get("authors", []))
        }

        for header, content in sections.items():
            if not content.strip(): continue
            chunk_id = f"{parent_id}#{header.replace(' ', '_')[:50]}"
            chunk_meta = {**base_meta, "section": header}

            docs.append(content.replace("\n", " "))
            metas.append(chunk_meta)
            ids.append(chunk_id)

        # 4. Embed (Domain: Embedder)
        if docs:
            vectors_np = embed_service.encode(docs, model_name=model_name)

            # 5. Store (Domain: Database)
            db_service.upsert_chunks(
                model_key=model_name,
                ids=ids,
                documents=docs,
                embeddings=vectors_np.tolist(),
                metadata=metas
            )

        return {
            "filename": file.filename,
            "message": "Ingestion successful",
            "chunks_added": len(docs),
            "parent_id": parent_id
        }

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/query", response_model=QueryResponse)
async def search(
        request: QueryRequest,
        embed_service: EmbeddingService = Depends(get_embedding_service),
        db_service: VectorDBService = Depends(get_db_service),
):
    """
    Search for relevant document chunks based on a query.
    1. Embed the query text.
    2. Search the vector database for nearest neighbors.
    3. Format and return the results.
    :param request:
    :param embed_service:
    :param db_service:
    :return:
    """
    # 1. Embed Query
    query_vec_np = embed_service.encode([request.query_text], model_name=request.model_name)

    # 2. Search DB
    results = db_service.query(
        model_key=request.model_name,
        query_embedding=query_vec_np[0].tolist(),
        n_results=request.n_results
    )

    # 3. Format Response
    formatted = []
    if results["ids"] and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            formatted.append(SearchResult(
                doc_id=results["ids"][0][i],
                score=results["distances"][0][i],
                content=results["documents"][0][i],
                metadata=results["metadatas"][0][i]
            ))

    return {"results": formatted}


@app.post("/rag", response_model=RagResponse)
async def rag_answer(
        request: RagRequest,
        embed_service: EmbeddingService = Depends(get_embedding_service),
        db_service: VectorDBService = Depends(get_db_service),
        rec_service: SemanticScholarService = Depends(get_recommendation_service),
):
    """
    Run the RAG pipeline using the existing embedding + Chroma services.
    """
    response = run_rag_answer(
        question=request.question,
        model_name=request.model_name,
        n_results=request.n_results,
        include_sources=request.include_sources,
        llm_model=request.llm_model,
        temperature=request.temperature,
        embed_service=embed_service,
        db_service=db_service,
    )

    sources = None
    if response.sources:
        sources = []
        for doc in response.sources:
            sources.append(SearchResult(
                doc_id=doc.metadata.get("id", "") if doc.metadata else "",
                score=doc.metadata.get("score", 0.0) if doc.metadata else 0.0,
                content=doc.page_content,
                metadata=doc.metadata or {}
            ))

    answer = response.answer
    template = response.template
    status = response.status
    needs_search = response.needs_search

    if response.needs_search:
        papers = []
        try:
            papers = await rec_service.search_papers(request.question, limit=1)
        except HTTPException:
            papers = []

        if papers:
            paper = papers[0]
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract") or "No abstract available."
            answer = (
                "Looking for additional scientific information online. "
                f"Top result: {title}. Abstract: {abstract}"
            )
        else:
            answer = "Looking for additional scientific information online, but no results were found."

        sources = None
        status = "ok"
        needs_search = False
        template = "online"

    return {
        "answer": answer,
        "template": template,
        "status": status,
        "needs_search": needs_search,
        "sources": sources
    }


@app.post("/delete", response_model=StatusResponse)
async def delete_documents(
        request: DeleteRequest,
        db_service: VectorDBService = Depends(get_db_service),
):
    """
    Delete specific chunks by their IDs.
    """
    try:
        db_service.delete_chunks(request.model_name, request.doc_ids)
        return {
            "status": "success",
            "message": f"Deleted {len(request.doc_ids)} chunks from {request.model_name}",
            "count": len(request.doc_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset", response_model=StatusResponse)
async def reset_collection(
        request: ResetRequest,
        db_service: VectorDBService = Depends(get_db_service),
):
    """
    DANGER: Completely wipes the vector database for the selected model.
    """
    try:
        db_service.clear_collection(request.model_name)
        return {
            "status": "success",
            "message": f"All data in '{request.model_name}' has been wiped.",
            "count": 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_database_stats(
        model_name: str = "bert",  # Defaults to 'bert' if not provided
        db_service: VectorDBService = Depends(get_db_service)
):
    """
    Returns the total number of chunks for the specified model's collection.
    """
    try:
        count = db_service.get_stats(model_name)
        return {
            "model_name": model_name,
            "count": count
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inspect", response_model=InspectResponse)
async def inspect_document(
        doc_id: str,
        model_name: str = "bert",
        db_service: VectorDBService = Depends(get_db_service)
):
    """
    Retrieve raw text/metadata.
    Usage: /inspect?doc_id=some_id&model_name=bert
    """
    try:
        data = db_service.get_chunk(model_name, doc_id)
        if not data:
            raise HTTPException(status_code=404, detail=f"ID '{doc_id}' not found in model '{model_name}'")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/list-ids", response_model=IDListResponse)
async def list_document_ids(
        model_name: str = "bert",
        limit: int = 100,
        offset: int = 0,
        db_service: VectorDBService = Depends(get_db_service)
):
    """
    Get a list of IDs to use with /inspect or /delete.
    """
    try:
        ids = db_service.list_ids(model_name, limit, offset)
        return {
            "model_name": model_name,
            "ids": ids,
            "total_in_batch": len(ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/embed", response_model=EmbedDebugResponse)
async def debug_embedding(
        request: EmbedDebugRequest,
        embed_service: EmbeddingService = Depends(get_embedding_service)
):
    """
    Generate an embedding for a raw text string.
    Useful to check vector dimensions or determinism.
    """
    try:
        # encode returns a numpy array, usually shape (1, dim) for a single string
        vector_np = embed_service.encode([request.text], model_name=request.model_name)

        # Flatten to a simple list
        vector_list = vector_np[0].tolist()

        return {
            "model_name": request.model_name,
            "vector_preview": vector_list,
            "dimension": len(vector_list)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/parse-pdf")
async def debug_pdf_parsing(
        file: UploadFile = File(...),
        pdf_service: PDFProcessorService = Depends(get_pdf_service)
):
    """
    Parses a PDF and returns the raw JSON (Metadata + Sections).
    Does NOT save to ChromaDB. Use this to test Docling extraction.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, "File must be a PDF")

    # 1. Save to temp file (Reusing logic from ingest)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 2. Run Docling Processing
        metadata, sections = pdf_service.process_pdf(tmp_path)

        # 3. Return Raw Data
        return {
            "filename": file.filename,
            "metadata_extracted": metadata,
            "section_count": len(sections),
            "sections": sections
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_papers(
        request: RecommendationRequest,
        rec_service: SemanticScholarService = Depends(get_recommendation_service)
):
    """
    Get paper recommendations based on positive and negative examples
    using the Semantic Scholar API.
    """
    try:
        results = await rec_service.get_recommendations(
            positive_ids=request.positive_paper_ids,
            negative_ids=request.negative_paper_ids,
            limit=request.limit
        )
        return {"recommendations": results}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/paper/search", response_model=PaperSearchResponse)
async def search_paper_id_endpoint(
        query: str,
        rec_service: SemanticScholarService = Depends(get_recommendation_service)
):
    """
    Search for a paper ID using its title or a search query.
    """
    try:
        paper_id = await rec_service.search_paper_id(query)
        return {"query": query, "paperId": paper_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
