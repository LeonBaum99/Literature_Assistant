from typing import List, Dict

import chromadb


class VectorDBService:
    def __init__(self, db_path: str, collection_names: Dict[str, str]):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection_names = collection_names
        # Cache collections to avoid fetching them repeatedly
        self._collections = {}

    def get_collection(self, model_key: str) -> chromadb.Collection:
        if model_key not in self._collections:
            name = self.collection_names.get(model_key, "default_collection")
            self._collections[model_key] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "ip"}
            )
        return self._collections[model_key]

    def upsert_chunks(
        self,
        model_key: str,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadata: List[Dict]
    ):
        collection = self.get_collection(model_key)
        collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadata)

    def query(self, model_key: str, query_embedding: List[float], n_results: int):
        collection = self.get_collection(model_key)
        return collection.query(query_embeddings=[query_embedding], n_results=n_results)

    def delete_chunks(self, model_key: str, ids: List[str]):
        """Deletes specific chunks by ID."""
        collection = self.get_collection(model_key)
        collection.delete(ids=ids)

    def clear_collection(self, model_key: str):
        """
        Deletes the entire collection from the DB and clears the local cache.
        """
        # 1. Get the actual name (e.g., "scientific_papers_bert")
        name = self.collection_names.get(model_key)
        if not name:
            return

        # 2. Delete via the Client (not the collection object)
        try:
            self.client.delete_collection(name)
        except ValueError:
            # Chroma raises ValueError if collection doesn't exist, which is fine
            pass

        # 3. IMPORTANT: Invalidating the cache
        # If we don't do this, self.get_collection() will return a dead object
        if model_key in self._collections:
            del self._collections[model_key]

        print(f"Collection '{name}' deleted and cache cleared.")

    def get_stats(self, model_key: str) -> int:
        """Returns the count of documents for a specific collection."""
        name = self.collection_names.get(model_key)
        if not name:
            raise ValueError(f"Unknown model key: {model_key}")

        try:
            coll = self.client.get_collection(name)
            return coll.count()
        except ValueError:
            return 0

    def get_chunk(self, model_key: str, doc_id: str):
        """Fetches a single chunk's data by ID."""
        collection = self.get_collection(model_key)
        result = collection.get(ids=[doc_id])

        if not result["ids"]:
            return None

        return {
            "id": result["ids"][0],
            "document": result["documents"][0],
            "metadata": result["metadatas"][0] if result["metadatas"] else {}
        }

    def list_ids(self, model_key: str, limit: int = 100, offset: int = 0):
        """Lists IDs from a collection with pagination."""
        collection = self.get_collection(model_key)
        result = collection.get(limit=limit, offset=offset, include=[])
        return result["ids"]
