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
        collection = self.get_collection(model_key)
        collection.delete(ids=ids)

    def clear_collection(self, model_key: str):
        collection = self.get_collection(model_key)
        collection.delete()
