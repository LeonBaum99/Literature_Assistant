import httpx
from typing import List, Dict, Any, Optional
from fastapi import HTTPException


class SemanticScholarService:

    RECOMMENDATION_URL = "https://api.semanticscholar.org/recommendations/v1/papers/"
    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, api_key: str = None):
        self.api_key = api_key
        if not self.api_key:
            print("⚠️ Warning: No Semantic Scholar API Key provided. Rate limits will be strict.")

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

        # Query Params: Contains 'limit' and 'fields'
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
                print(f"❌ {error_msg}")
                raise HTTPException(status_code=e.response.status_code, detail=error_msg)
            except Exception as e:
                print(f"❌ Recommendation Service Failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def search_papers(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,year,url,authors,abstract"
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
                    raise HTTPException(status_code=403, detail="Semantic Scholar API Key invalid or missing.")

                response.raise_for_status()
                data = response.json()

                return data.get("data", [])

            except httpx.HTTPStatusError as e:
                error_msg = f"Semantic Scholar Error {e.response.status_code}: {e.response.text}"
                print(f"Semantic Scholar Search Error: {error_msg}")
                raise HTTPException(status_code=e.response.status_code, detail=error_msg)
            except Exception as e:
                print(f"Paper Search Failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

    async def search_paper_id(self, query: str, limit: int = 1) -> Optional[str]:
        """
        Searches for a paper by title/query and returns the first matching paperId.
        Does NOT raise HTTP exceptions, just returns None on failure/empty.
        """
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title"
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
                    print("❌ Semantic Scholar API Key invalid or missing during search.")
                    return None

                response.raise_for_status()
                data = response.json()

                results = data.get("data", [])

                if results and len(results) > 0:
                    # Return the ID of the first match
                    print(f"✅ Found Paper ID: {results[0].get('paperId')} for query: '{query}'")
                    return results[0].get("paperId")

                print(f"⚠️ No paper found for query: '{query}'")
                return None

            except Exception as e:
                print(f"❌ Paper Search Failed: {e}")
                return None
