"""
Retrieval layer for the RAG pipeline.
This module handles similarity search using the vector store.
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def _load_module(module_name: str, file_name: str):
    file_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vector_db_module = _load_module("vector_db_mod", "5.vector_db.py")
embedding_module = _load_module("embedding_mod", "4.embedding.py")

VectorStore = vector_db_module.VectorStore
EmbeddingModel = embedding_module.EmbeddingModel


def retrieve_similar_chunks(
    query: str,
    vector_store: VectorStore,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    top_k: int = 5,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> Sequence[Dict[str, Any]]:
    model = EmbeddingModel(model_name)
    query_embedding = model.embed_text(query)
    return vector_store.search(query_embedding, top_k=top_k, metadata_filter=metadata_filter)
