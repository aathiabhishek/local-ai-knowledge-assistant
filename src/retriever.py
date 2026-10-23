import re
from typing import List, Dict, Optional

import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

_TOKEN_PATTERN = re.compile(r"\b\w+\b")


def _tokenize(text: str) -> List[str]:
    """Strip punctuation so words like 'FAISS.' and 'FAISS' match in BM25."""
    return _TOKEN_PATTERN.findall(text.lower())


class HybridRetriever:
    """
    Two-Stage Retrieval Pipeline:
      - Stage 1: FAISS (Dense) + BM25 (Sparse) merged via Reciprocal Rank Fusion (RRF).
      - Stage 2: Cross-Encoder Reranker scores [Query, Chunk] pairs directly.
    """

    def __init__(
        self,
        embedding_model_name: str,
        reranker_model_name: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.reranker = (
            CrossEncoder(reranker_model_name) if reranker_model_name else None
        )
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

        _, dimension = embeddings.shape
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        tokenized = [_tokenize(text) for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or self.bm25 is None:
            return []

        n_docs = len(self.documents)
        candidate_k = min(max(top_k * 4, 15), n_docs)

        # -----------------------------
        # Stage 1A: Semantic Search (FAISS)
        # -----------------------------
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")

        _, semantic_ids_batch = self.index.search(query_embedding, candidate_k)
        first_query_ids = next(iter(semantic_ids_batch))

        semantic_rank = {
            int(doc_id): rank
            for rank, doc_id in enumerate(first_query_ids, start=1)
            if doc_id >= 0
        }

        # -----------------------------
        # Stage 1B: Keyword Search (BM25)
        # -----------------------------
        tokenized_query = _tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        keyword_ids = np.argsort(bm25_scores)[::-1][:candidate_k]

        keyword_rank = {
            int(doc_id): rank
            for rank, doc_id in enumerate(keyword_ids, start=1)
            if bm25_scores[doc_id] > 0.0
        }

        # -----------------------------
        # Stage 1C: Reciprocal Rank Fusion (RRF)
        # -----------------------------
        all_ids = set(semantic_rank) | set(keyword_rank)
        if not all_ids:
            return []

        rrf_k = 60.0
        fused = []
        for doc_id in all_ids:
            rrf_score = 0.0
            if doc_id in semantic_rank:
                rrf_score += 1.0 / (rrf_k + semantic_rank[doc_id])
            if doc_id in keyword_rank:
                rrf_score += 1.0 / (rrf_k + keyword_rank[doc_id])
            fused.append((rrf_score, doc_id))

        fused.sort(reverse=True)
        stage1_candidates = fused[:candidate_k]

        # -----------------------------
        # Stage 2: Cross-Encoder Reranking
        # -----------------------------
        if self.reranker is not None:
            pairs = [
                [query, self.documents[doc_id]["text"]]
                for _, doc_id in stage1_candidates
            ]
            rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

            reranked = [
                (float(rerank_score), float(rrf_score), doc_id)
                for (rrf_score, doc_id), rerank_score in zip(
                    stage1_candidates, rerank_scores
                )
            ]
            reranked.sort(reverse=True)

            results = []
            for rerank_score, rrf_score, doc_id in reranked[:top_k]:
                doc = dict(self.documents[doc_id])
                doc["score"] = rerank_score
                doc["rrf_score"] = rrf_score
                results.append(doc)
            return results

        # Fallback if reranker is None
        results = []
        for rrf_score, doc_id in stage1_candidates[:top_k]:
            doc = dict(self.documents[doc_id])
            doc["score"] = float(rrf_score)
            results.append(doc)

        return results
