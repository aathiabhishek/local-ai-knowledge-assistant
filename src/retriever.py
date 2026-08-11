from typing import List, Dict

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """
    Combines:
      - FAISS semantic retrieval
      - BM25 keyword retrieval

    A reciprocal-rank-fusion style score combines the two rankings.
    """

    def __init__(self, embedding_model_name: str):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.documents = []
        self.index = None
        self.bm25 = None

    def build(self, documents: List[Dict]):
        if not documents:
            raise ValueError("No documents were provided.")

        self.documents = documents

        texts = [doc["text"] for doc in documents]

        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        dimension = embeddings.shape[1]

        # Inner product on normalized vectors = cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        tokenized = [
            text.lower().split()
            for text in texts
        ]

        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or self.bm25 is None:
            return []

        n_docs = len(self.documents)
        candidate_k = min(max(top_k * 3, 10), n_docs)

        # -----------------------------
        # Semantic search
        # -----------------------------
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        semantic_scores, semantic_ids = self.index.search(
            query_embedding,
            candidate_k,
        )

        semantic_rank = {}
        for rank, doc_id in enumerate(semantic_ids[0], start=1):
            if doc_id >= 0:
                semantic_rank[int(doc_id)] = rank

        # -----------------------------
        # BM25 search
        # -----------------------------
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        keyword_ids = np.argsort(bm25_scores)[::-1][:candidate_k]

        keyword_rank = {
            int(doc_id): rank
            for rank, doc_id in enumerate(keyword_ids, start=1)
        }

        # -----------------------------
        # Reciprocal Rank Fusion
        # -----------------------------
        all_ids = set(semantic_rank) | set(keyword_rank)
        fused = []

        rrf_k = 60.0

        for doc_id in all_ids:
            score = 0.0

            if doc_id in semantic_rank:
                score += 1.0 / (rrf_k + semantic_rank[doc_id])

            if doc_id in keyword_rank:
                score += 1.0 / (rrf_k + keyword_rank[doc_id])

            fused.append((doc_id, score))

        fused.sort(key=lambda x: x[1], reverse=True)

        results = []

        for doc_id, score in fused[:top_k]:
            doc = dict(self.documents[doc_id])
            doc["score"] = float(score)
            results.append(doc)

        return results
