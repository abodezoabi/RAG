"""
Vector database layer for storing chunk embeddings and metadata.
This implementation uses an in-memory vector store with optional FAISS support.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover
    faiss = None


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if not vec_a or not vec_b:
        return 0.0

    if np is not None:
        a = np.asarray(vec_a, dtype=float)
        b = np.asarray(vec_b, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(x * x for x in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorStore:
    """A lightweight in-memory vector database."""

    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []

    def add(self, chunk: Dict[str, Any], embedding: Sequence[float]) -> None:
        self.documents.append(chunk)
        self.embeddings.append(list(embedding))

    def add_many(self, chunks: Sequence[Dict[str, Any]], embeddings: Sequence[Sequence[float]]) -> None:
        for chunk, embedding in zip(chunks, embeddings):
            self.add(chunk, embedding)

    def metadata_matches(self, metadata: Dict[str, Any], filter_dict: Optional[Dict[str, Any]]) -> bool:
        if not filter_dict:
            return True
        for key, value in filter_dict.items():
            if metadata.get(key) != value:
                return False
        return True

    def search(self, query_embedding: Sequence[float], top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.embeddings:
            return []

        ranked: List[Tuple[float, int]] = []
        for index, stored_embedding in enumerate(self.embeddings):
            metadata = self.documents[index].get("metadata", {})
            if not self.metadata_matches(metadata, metadata_filter):
                continue
            score = cosine_similarity(query_embedding, stored_embedding)
            ranked.append((score, index))

        ranked.sort(key=lambda item: item[0], reverse=True)
        top_indices = ranked[:top_k]

        results = []
        for _, index in top_indices:
            doc = self.documents[index].copy()
            doc["score"] = cosine_similarity(query_embedding, self.embeddings[index])
            results.append(doc)

        return results


def build_vector_store(chunks: Sequence[Dict[str, Any]], embeddings: Sequence[Sequence[float]]) -> VectorStore:
    store = VectorStore()
    store.add_many(chunks, embeddings)
    return store
