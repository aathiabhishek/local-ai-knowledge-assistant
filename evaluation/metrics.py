"""Source-passage relevance metrics, independent of answer-word overlap."""


def source_retrieval_metrics(relevant_sources, hits):
    """Each benchmark source file represents one original SQuAD paragraph.

    A hit identifies a relevant parent passage, not necessarily an answer-bearing
    chunk. Multiple chunks from one passage retain their actual retrieval ranks.
    """
    relevant = set(relevant_sources)
    if not relevant:
        raise ValueError("Answerable questions require relevant source IDs.")
    rank = next((i for i, hit in enumerate(hits, 1)
                 if hit.get("source") in relevant), None)
    return {
        "source_hit_at_k": float(rank is not None),
        "source_reciprocal_rank": 1.0 / rank if rank is not None else 0.0,
    }
