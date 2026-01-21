import asyncio
import json
from pathlib import Path
from typing import Callable, Any, List, Dict, Optional

from backend.services.rag_answer_service import ChromaRagRetriever
from backend.services.recommendation import SemanticScholarService
from backend.services.vector_db import VectorDBService
from llmAG.rag import RagPipeline
from pdfProcessing.docling_PDF_processor import DoclingPDFProcessor
from zotero_integration.zotero_client import ZoteroClient


def perform_online_search_sync(
        service: SemanticScholarService,
        query: str,
        top_k: int
) -> str:
    """
    Wrapper to run smart_search synchronously.
    It loops through ALL retrieved papers and formats them into a single string,
    including the Title, Year, URL, and Abstract.
    """
    try:
        papers = asyncio.run(service.smart_search(query, limit=top_k))

        if not papers:
            return "Looking for additional scientific information online, but no results were found.\n"

        # Loop through results and build the answer string
        results_text = [f"Found {len(papers)} online sources for context:\n"]

        for i, paper in enumerate(papers, 1):
            title = paper.get("title", "Unknown Title")
            year = paper.get("year", "N/A")
            url = paper.get("url", "No URL available")

            abstract = paper.get("abstract") or "No abstract available."
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."

            # Format the entry with the URL included
            entry = (
                f"{'=' * 80}\n"
                f"[Online Source {i}]\n"
                f"{'=' * 80}\n"
                f"{title} ({year})\n"
                f"Link: {url}\n"
                f"Abstract: {abstract}\n"
                f"{'=' * 80}\n"
            )
            results_text.append(entry)

        return "\n".join(results_text)

    except Exception as e:
        print(f"Error during Semantic Scholar search: {e}")
        return "An error occurred while searching for online sources."


def query_rag(
        rag_pipeline: RagPipeline,
        retriever: ChromaRagRetriever,
        rec_service: SemanticScholarService,
        question: str,
        top_k: int = 5,
        show_context: bool = False,
        show_sources: bool = True,
        search_for_new_context: bool = False,
        top_k_results: int = 1
):
    """
    Main function to query the RAG pipeline with optional online search fallback.
    """
    if rag_pipeline is None:
        print("✗ RAG pipeline not available")
        return None

    print(f"\n{'=' * 80}")
    print(f"Query: {question}")
    print(f"{'=' * 80}\n")

    # 1. Retrieve context
    retrieved_docs = retriever.get_relevant_documents(question, k=top_k)
    print(f"Retrieved {len(retrieved_docs)} chunks\n")

    if show_context:
        print(f"{'=' * 80}")
        print("CONTEXT")
        print(f"{'=' * 80}")
        for i, doc in enumerate(retrieved_docs):
            print(f"\n[{i + 1}] {doc.metadata.get('section', 'N/A')}")
            print(f"{doc.page_content[:200]}...\n")

    # 2. Generate answer
    try:
        response = rag_pipeline.run(question, k=top_k, include_sources=True)

        print(f"{'=' * 80}")
        print("ANSWER")
        print(f"{'=' * 80}\n")
        print(response.answer)

        if show_sources and hasattr(response, 'sources'):
            print(f"\n{'=' * 80}")
            print("SOURCES")
            print(f"{'=' * 80}")
            for i, source in enumerate(response.sources):
                title = source.metadata.get('title', 'Unknown')
                # Truncate title if it's too long
                if len(title) > 60:
                    title = title[:60] + "..."

                print(f"\n[{i + 1}] {title}")
                print(f"    Section: {source.metadata.get('section', 'N/A')}")
            print(f"{'=' * 80}\n")

        # 3. Handle async search synchronously via Wrapper
        if search_for_new_context and getattr(response, 'needs_search', False):
            print(f"DEBUG: Triggering online search for {top_k_results} papers...")

            new_answer_text = perform_online_search_sync(
                service=rec_service,
                query=question,
                top_k=top_k_results
            )

            response.answer = new_answer_text

            print(f"{'=' * 80}")
            print("NEW ANSWER (Enhanced with Online Search)")
            print(f"{'=' * 80}\n")
            print(response.answer)

        return response

    except Exception as e:
        print(f"✗ Error: {e}")
        return None


def ingest_pdf(
        pdf_path: Path,
        processor: DoclingPDFProcessor,
        db_service: VectorDBService,
        embedder: Any,
        create_chunks_func: Callable,
        model_key: str = "bert",
        zotero_loader: ZoteroClient = None,
        max_chunk_size: int = 500,
        overlap_size: int = 50
) -> int:
    """
    Ingest single PDF: Process → Chunk → Embed → Store.

    Args:
        pdf_path: Path to the PDF file.
        processor: Service to process/read PDF content.
        db_service: Service to interact with the Vector DB.
        embedder: Service to encode text into embeddings.
        create_chunks_func: Function to split text sections into chunks.
        model_key: Key for the model version in the DB (default 'bert').
        zotero_loader: Optional service to fetch Zotero metadata.
        max_chunk_size: Maximum characters per chunk.
        overlap_size: Overlap characters between chunks.

    Returns:
        int: Number of chunks ingested.
    """
    print(f"\nProcessing: {pdf_path.name}")

    # Try Zotero metadata first
    zotero_meta = None
    if zotero_loader:
        zotero_meta = zotero_loader.get_metadata_by_filename(pdf_path.name)
        if zotero_meta:
            print(f"  Using Zotero metadata: '{zotero_meta['title'][:50]}...'")
        else:
            print(f"  Warning: No Zotero match - using Docling extraction")

    # Process PDF
    # Note: Ensure the processor passed in has a process_pdf method
    metadata, sections = processor.process_pdf(str(pdf_path), zotero_metadata=zotero_meta)
    print(f"  Extracted {len(sections)} sections")

    # Create chunks
    docs, metas, ids = create_chunks_func(
        filename=pdf_path.name,
        metadata=metadata,
        sections=sections,
        max_chunk_size=max_chunk_size,
        overlap_size=overlap_size
    )
    print(f"  Created {len(docs)} chunks")

    if not docs:
        print("  Error: No chunks created")
        return 0

    # Embed and store
    embeddings = embedder.encode(docs)
    db_service.upsert_chunks(
        model_key=model_key,
        ids=ids,
        documents=docs,
        embeddings=embeddings.tolist(),
        metadata=metas
    )

    print(f"  Ingested {len(docs)} chunks")
    return len(docs)


def load_eval_dataset(filename: str = "eval_dataset.json") -> List[Dict[str, Any]]:
    """
    Loads evaluation dataset from current or parent directory.
    """
    potential_dirs = [Path.cwd(), Path.cwd().parent]
    for directory in potential_dirs:
        file_path = directory / filename
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                print(f"Loaded {len(data)} questions from {file_path}")
                return data
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from {file_path}: {e}")
                return []

    print(f"Warning: {filename} not found in {potential_dirs}")
    return []


def show_llm_prompt(
        rag_pipeline: RagPipeline,
        retriever: ChromaRagRetriever,
        question: str,
        top_k: int = 5,
        template_name: str = "answer"
):
    """
    Display the exact prompt that will be sent to the LLM.
    """
    if not rag_pipeline or not retriever:
        print("Error: Pipeline or Retriever not initialized.")
        return

    retrieved_docs = retriever.get_relevant_documents(question, k=top_k)
    context = rag_pipeline._format_context(retrieved_docs)

    # Retrieve template safely with fallback
    prompt_template = rag_pipeline._prompts.get(template_name)
    if not prompt_template:
        prompt_template = rag_pipeline._prompts.get("answer")

    if not prompt_template:
        print(f"Error: Template '{template_name}' not found.")
        return

    formatted_prompt = prompt_template.format_messages(question=question, context=context)

    print(f"{'=' * 80}")
    print(f"EXACT PROMPT SENT TO LLM")
    print(f"{'=' * 80}")
    print(f"Template: {template_name} | Retrieved chunks: {len(retrieved_docs)} | Context: {len(context)} chars\n")

    for i, msg in enumerate(formatted_prompt):
        role = msg.__class__.__name__.replace('Message', '').upper()
        print(f"\n{'=' * 80}")
        print(f"MESSAGE {i + 1}: {role}")
        print(f"{'=' * 80}\n")
        print(msg.content)

    print(f"\n{'=' * 80}")
    print(f"Total prompt length: {sum(len(m.content) for m in formatted_prompt)} chars")
    print(f"{'=' * 80}")


def log_retrieval_results(
        results: Dict[str, Any],
        query: str,
        output_file: Optional[Path] = None
) -> str:
    """
    Formats retrieval results, prints them to console, and optionally saves to a file.
    """
    # Initialize output with header
    output_lines = [f"QUERY: {query}\n", "=" * 80 + "\nRETRIEVAL RESULTS\n" + "=" * 80 + "\n"]

    # Check if results exist
    if not results.get('ids') or not results['ids'][0]:
        msg = "No results found."
        print(msg)
        return msg

    # Loop through results
    for i in range(len(results['ids'][0])):
        chunk_id = results['ids'][0][i]
        distance = results['distances'][0][i]
        content = results['documents'][0][i]
        # Handle None metadata safely
        meta = results['metadatas'][0][i] if results['metadatas'][0][i] else {}

        chunk_output = f"""
{'=' * 80}
Rank {i + 1} | Distance: {distance:.4f}
{'=' * 80}
ID:      {chunk_id}
Section: {meta.get('section', 'N/A')}
Paper:   {meta.get('title', 'N/A')}
Authors: {meta.get('authors', 'N/A')}

Content ({len(content)} chars):
{'-' * 80}
{content}
"""
        print(chunk_output)
        output_lines.append(chunk_output)

    full_output = "\n".join(output_lines)

    # Save to file if path provided
    if output_file:
        try:
            if not output_file.parent.exists():
                output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(full_output)
            print(f"\nFull retrieval output saved to {output_file}")
        except Exception as e:
            print(f"Error saving to file: {e}")

    return full_output
