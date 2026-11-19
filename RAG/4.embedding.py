"""
Embedding layer for the RAG pipeline.
Responsible for turning chunked text into vectors.
"""

from typing import List, Sequence

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    """Simple wrapper around a sentence-transformer model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> List[float]:
        if not text or not text.strip():
            return []
        return self.model.encode(text, convert_to_numpy=True).tolist()

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            return []
        embeddings = self.model.encode(valid_texts, convert_to_numpy=True)
        return embeddings.tolist()


def embed_text(text: str, model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[float]:
    model = EmbeddingModel(model_name)
    return model.embed_text(text)


def embed_chunks(chunks: Sequence[str], model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[List[float]]:
    model = EmbeddingModel(model_name)
    return model.embed_texts(chunks)
