import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException

from backend.dependencies import get_db_service, get_embedding_service, get_pdf_service
# Import Internal Modules
from backend.schemas import (
    QueryRequest, QueryResponse, IngestResponse, SearchResult,
    DeleteRequest, ResetRequest, StatusResponse
)
from backend.services.embedder import EmbeddingService
from backend.services.processor import PDFProcessorService
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
        parent_id = metadata.get("arxiv_id", file.filename).replace(" ", "_")
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
