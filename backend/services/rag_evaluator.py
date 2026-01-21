import time
from typing import List, Dict, Any, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EnhancedRAGEvaluator:
    """
    Service for evaluating RAG pipeline performance across various metrics
    including chunk retrieval, paper recall, and semantic answer quality.
    """

    def __init__(self, pipeline: Any, model_name: str = 'all-MiniLM-L6-v2'):
        self.pipeline = pipeline
        self.results: List[Dict[str, Any]] = []

        print(f"Loading semantic similarity model: {model_name}...")
        try:
            self.semantic_model = SentenceTransformer(model_name)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def evaluate(self, dataset: List[Dict[str, Any]], top_k: int = 5) -> pd.DataFrame:
        """
        Main entry point to evaluate a dataset against the RAG pipeline.

        Args:
            dataset: List of dictionaries containing questions and ground truths.
            top_k: Number of contexts to retrieve.

        Returns:
            pd.DataFrame containing detailed evaluation metrics.
        """
        print(f"Starting evaluation of {len(dataset)} questions...")
        self.results = []

        for item in tqdm(dataset):
            result = self._process_single_item(item, top_k)
            self.results.append(result)

        return pd.DataFrame(self.results)

    def _process_single_item(self, item: Dict[str, Any], top_k: int) -> Dict[str, Any]:
        """
        Processes a single evaluation item: runs pipeline, calculates metrics, handles errors.
        """
        question = item['question']
        target_tag = item.get('target_tag')
        tier = item.get('tier')

        # Prepare safe defaults for the result row
        result_row = {
            "Tier": tier,
            "Question": self._truncate_text(question, 60),
            "Target_Tag": target_tag,
            "Exact_Chunk_Match": False if item.get('expected_chunk_id') else None,
            "Chunk_Rank": None,
            "Semantic_Chunk_Hit": None,
            "Best_Chunk_Similarity": None,
            "Num_Papers": 0,
            "Multi_Paper_Match": False,
            "Paper_Recall": None,
            "Paper_Precision": None,
            "Answer_Similarity": None,
            "Papers": "ERROR",
            "Latency": 0.0
        }

        start_time = time.time()
        try:
            # 1. Run Pipeline
            response = self.pipeline.run(question, k=top_k, include_sources=True)
            elapsed = time.time() - start_time

            # 2. Extract Basic Info
            retrieved_sources = response.sources
            retrieved_filenames = [src.metadata.get('filename', '') for src in retrieved_sources]
            unique_papers = list(set(retrieved_filenames))

            # 3. Calculate Metrics
            chunk_metrics = self._calculate_chunk_metrics(
                item.get('expected_chunk_id'),
                retrieved_sources
            )

            paper_metrics = self._calculate_paper_metrics(
                item.get('expected_papers', []),
                retrieved_filenames
            )

            answer_metrics = self._calculate_answer_quality(
                item.get('expected_answer'),
                response.answer
            )

            # 4. Update Result Row
            result_row.update({
                "Exact_Chunk_Match": chunk_metrics["exact_match"],
                "Chunk_Rank": chunk_metrics["rank"],
                "Semantic_Chunk_Hit": chunk_metrics["semantic_hit"],
                "Best_Chunk_Similarity": chunk_metrics["similarity"],

                "Num_Papers": len(unique_papers),
                "Multi_Paper_Match": len(unique_papers) >= 2 if tier == 3 else None,
                "Paper_Recall": paper_metrics["recall"],
                "Paper_Precision": paper_metrics["precision"],

                "Answer_Similarity": answer_metrics["similarity"],

                "Papers": " | ".join([p.split(' - ')[0][:30] for p in unique_papers[:2]]),
                "Latency": round(elapsed, 2)
            })

        except Exception as e:
            print(f"Error evaluating question '{self._truncate_text(question, 30)}': {e}")
            # Returns the safe default row created at start of method

        return result_row

    def _calculate_chunk_metrics(self, expected_chunk_id: Optional[str], sources: List[Any]) -> Dict[str, Any]:
        """
        Determines if the correct chunk was retrieved (Exact Match) or if a semantically
        similar chunk was found (Semantic Hit).
        """
        metrics = {
            "exact_match": None,
            "rank": None,
            "semantic_hit": None,
            "similarity": None
        }

        if not expected_chunk_id:
            return metrics

        metrics["exact_match"] = False

        # 1. Check Exact Match
        for rank, src in enumerate(sources, 1):
            parent_id = src.metadata.get('parent_id', '')
            # Match strict ID or ID before fragment hash
            if parent_id == expected_chunk_id or expected_chunk_id.split('#')[0] in parent_id:
                metrics["exact_match"] = True
                metrics["rank"] = rank
                break

        # 2. Check Semantic Similarity (if exact match failed)
        if not metrics["exact_match"]:
            try:
                # Retrieve the actual text of the expected chunk from the DB
                collection = self.pipeline.retriever.db_service.get_collection(
                    self.pipeline.retriever.model_name
                )
                expected_docs = collection.get(ids=[expected_chunk_id])

                if expected_docs and expected_docs['documents']:
                    expected_text = expected_docs['documents'][0]
                    expected_embedding = self.semantic_model.encode([expected_text])

                    retrieved_texts = [src.page_content for src in sources]
                    if retrieved_texts:
                        retrieved_embeddings = self.semantic_model.encode(retrieved_texts)
                        similarities = cosine_similarity(expected_embedding, retrieved_embeddings)[0]

                        best_sim = float(similarities.max())
                        metrics["similarity"] = round(best_sim, 3)
                        metrics["semantic_hit"] = best_sim > 0.7
            except Exception as e:
                # If we can't fetch the expected chunk from DB, we skip semantic check
                pass

        return metrics

    def _calculate_paper_metrics(self, expected_papers: List[str], retrieved_filenames: List[str]) -> Dict[str, float]:
        """
        Calculates Precision and Recall for retrieved papers vs expected papers.
        """
        metrics = {"recall": None, "precision": None}

        if not expected_papers:
            return metrics

        retrieved_norm = {f.lower() for f in retrieved_filenames if f}
        expected_norm = {p.lower() for p in expected_papers}

        if not expected_norm:
            return metrics

        correct_papers = retrieved_norm & expected_norm

        if len(expected_norm) > 0:
            metrics["recall"] = round(len(correct_papers) / len(expected_norm), 3)

        if len(retrieved_norm) > 0:
            metrics["precision"] = round(len(correct_papers) / len(retrieved_norm), 3)

        return metrics

    def _calculate_answer_quality(self, expected_answer: Optional[str], generated_answer: str) -> Dict[str, float]:
        """
        Calculates semantic similarity between the generated answer and the ground truth.
        """
        metrics = {"similarity": None}

        if expected_answer and generated_answer:
            try:
                ans_embedding = self.semantic_model.encode([generated_answer])
                exp_embedding = self.semantic_model.encode([expected_answer])
                sim = float(cosine_similarity(ans_embedding, exp_embedding)[0][0])
                metrics["similarity"] = round(sim, 3)
            except Exception:
                pass

        return metrics

    @staticmethod
    def _truncate_text(text: str, length: int) -> str:
        """Helper to truncate strings for log display."""
        if not text:
            return ""
        return text[:length] + "..." if len(text) > length else text