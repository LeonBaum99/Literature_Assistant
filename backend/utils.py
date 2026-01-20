import asyncio

from backend.services.rag_answer_service import ChromaRagRetriever
from backend.services.recommendation import SemanticScholarService
from llmAG.rag import RagPipeline


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
