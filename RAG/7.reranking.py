"""
Reranking layer for RAG.
This module reorders retrieved chunks using a lightweight heuristic.
"""

from typing import Any, Dict, List, Sequence


def keyword_overlap_score(query: str, text: str) -> float:
    query_terms = set(query.lower().split())
    text_terms = set(text.lower().split())
    if not query_terms:
        return 0.0
    overlap = query_terms & text_terms
    return len(overlap) / len(query_terms)


def rerank_chunks(query: str, chunks: Sequence[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
    """Heuristic reranker: boosts chunks containing query terms."""
    scored = []
    for chunk in chunks:
        text = chunk.get("text", "")
        semantic_score = chunk.get("score", 0.0)
        overlap_score = keyword_overlap_score(query, text)
        total_score = semantic_score + overlap_score
        doc = chunk.copy()
        doc["rerank_score"] = total_score
        scored.append(doc)

    scored.sort(key=lambda item: item["rerank_score"], reverse=True)
    return scored[:top_n]
