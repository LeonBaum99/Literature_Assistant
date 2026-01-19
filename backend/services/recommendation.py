import httpx
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import HTTPException


class SemanticScholarService:
    RECOMMENDATION_URL = "https://api.semanticscholar.org/recommendations/v1/papers/"
    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    SNIPPET_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/snippet/search"
    BATCH_DETAILS_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if not self.api_key:
            print("Warning: No Semantic Scholar API Key provided. Rate limits will be strict.")

    async def get_recommendations(
            self,
            positive_ids: List[str],
            negative_ids: List[str] = None,
            limit: int = 10
    ) -> List[Dict[str, Any]]:

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        payload = {
            "positivePaperIds": positive_ids,
            "negativePaperIds": negative_ids or [],
        }

        params = {
            "fields": "paperId,title,year,url,authors,abstract",
            "limit": limit
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.RECOMMENDATION_URL,
                    json=payload,
                    headers=headers,
                    params=params,
                    timeout=10.0
                )

                if response.status_code == 403:
                    raise HTTPException(status_code=403, detail="Semantic Scholar API Key invalid or missing.")

                response.raise_for_status()
                data = response.json()

                return data.get("recommendedPapers", [])

            except httpx.HTTPStatusError as e:
                error_msg = f"Semantic Scholar Error {e.response.status_code}: {e.response.text}"
                print(f"Error: {error_msg}")
                raise HTTPException(status_code=e.response.status_code, detail=error_msg)
            except Exception as e:
                print(f"Error: Recommendation Service Failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def search_paper_ids(self, query: str, limit: int = 1) -> List[str]:
        """
        Searches for papers by query and returns a list of paperIds.
        Used as the first step before batch retrieval.
        """
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId"  # Only fetch ID to keep it lightweight
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.SEARCH_URL,
                    headers=headers,
                    params=params,
                    timeout=10.0
                )

                if response.status_code == 403:
                    print("Error: Semantic Scholar API Key invalid or missing during search.")
                    return []

                response.raise_for_status()
                data = response.json()
                results = data.get("data", [])

                ids = [item.get("paperId") for item in results if item.get("paperId")]

                if ids:
                    print(f"Found {len(ids)} Paper IDs for query: '{query}'")
                else:
                    print(f"No papers found for query: '{query}'")

                return ids

            except Exception as e:
                print(f"Error: Paper Search Failed: {e}")
                return []

    async def search_paper_id(self, query: str, limit: int = 1) -> Optional[str]:
        """
        Legacy/Convenience method: Returns the first paperId found.
        """
        ids = await self.search_paper_ids(query, limit)
        return ids[0] if ids else None

    async def search_text_snippets(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        if limit > 1000:
            limit = 1000

        params = {
            "query": query,
            "limit": limit
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    self.SNIPPET_SEARCH_URL,
                    headers=headers,
                    params=params,
                    timeout=10.0
                )

                if response.status_code == 403:
                    raise HTTPException(status_code=403, detail="Semantic Scholar API Key invalid or missing.")

                response.raise_for_status()
                data = response.json()
                return data.get("data", [])

            except httpx.HTTPStatusError as e:
                error_msg = f"Semantic Scholar Snippet Error {e.response.status_code}: {e.response.text}"
                print(f"Error: {error_msg}")
                raise HTTPException(status_code=e.response.status_code, detail=error_msg)
            except Exception as e:
                print(f"Error: Snippet Search Failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def get_papers_details_batch(
            self,
            paper_ids: List[str],
            fields: str = "paperId,title,year,url,authors,abstract"
    ) -> List[Dict[str, Any]]:
        """
        Get details for multiple papers at once using the batch endpoint.
        """
        if not paper_ids:
            return []

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "fields": fields
        }

        payload = {
            "ids": paper_ids
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    self.BATCH_DETAILS_URL,
                    params=params,
                    json=payload,
                    headers=headers,
                    timeout=10.0
                )

                if response.status_code == 403:
                    raise HTTPException(status_code=403, detail="Semantic Scholar API Key invalid or missing.")

                response.raise_for_status()
                data = response.json()

                # Filter out None results
                valid_papers = [p for p in data if p is not None]
                return valid_papers

            except httpx.HTTPStatusError as e:
                error_msg = f"Semantic Scholar Batch Error {e.response.status_code}: {e.response.text}"
                print(f"Error: {error_msg}")
                raise HTTPException(status_code=e.response.status_code, detail=error_msg)
            except Exception as e:
                print(f"Error: Batch Details Failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def smart_search(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        """
        1. Searches using standard paper search.
        2. If no results, searches text snippets.
        3. Extracts TITLES from snippets.
        4. Searches those titles sequentially with DELAYS to get valid Paper IDs.
        5. Uses those IDs to fetch full details via batch.
        """
        # 1. Try standard paper search
        print(f"Searching papers for: '{query}'")
        try:
            papers = await self.search_papers(query, limit=limit)
            if papers:
                print(f"Found {len(papers)} papers via standard search.")
                return papers
        except Exception as e:
            print(f"Error during Semantic Scholar search: {e}")

        # 2. Fallback: Snippet Search
        print("No papers found via standard search. Trying text snippets...")
        try:
            snippets = await self.search_text_snippets(query, limit=limit)

            if not snippets:
                print("No snippets found.")
                return []

            # 3. Extract Titles from snippets
            titles_to_search = set()
            for snippet in snippets:
                if "paper" in snippet and "title" in snippet["paper"]:
                    titles_to_search.add(snippet["paper"]["title"])
                elif "title" in snippet:
                    titles_to_search.add(snippet["title"])

            if not titles_to_search:
                print("Snippets found, but no titles could be extracted.")
                return []

            print(f"Extracted {len(titles_to_search)} titles from snippets. Verifying IDs with delays...")

            # 4. Search for IDs using the extracted Titles (Sequential with Sleep)
            found_ids = set()
            for i, title in enumerate(titles_to_search):
                try:
                    # RATE LIMIT PROTECTION:
                    # Sleep for 1 second between requests to avoid "Too Many Requests"
                    if i > 0:
                        await asyncio.sleep(1.0)

                    ids = await self.search_paper_ids(title, limit=1)
                    if ids:
                        found_ids.update(ids)
                except Exception as e:
                    print(f"Error resolving ID for title '{title}': {e}")
                    continue

            if not found_ids:
                print("No valid paper IDs found from snippet titles.")
                return []

            print(f"Found {len(found_ids)} confirmed Paper IDs. Fetching full details...")

            # Short delay before the final batch call
            await asyncio.sleep(0.5)

            # 5. Get full details for the confirmed IDs
            papers_data = await self.get_papers_details_batch(
                paper_ids=list(found_ids),
                fields="paperId,title,year,url,authors,abstract"
            )

            print(f"Retrieved details for {len(papers_data)} papers.")
            return papers_data

        except Exception as e:
            print(f"Error during Snippet/Title search sequence: {e}")
            return []