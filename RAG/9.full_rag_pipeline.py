"""
End-to-end RAG setup for a PDF-based knowledge system.
This file ties together the full process:
- document loading and parsing
- preprocessing
- chunking
- embedding
- vector store
- retrieval
- reranking
- evaluation / hallucination guardrails
"""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_module(module_name: str, file_name: str):
    file_path = Path(__file__).with_name(file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pdf_extractor = _load_module("pdf_extractor_mod", "1.pdf_extractor.py")
preprocessor = _load_module("preprocessor_mod", "2.preprocessing.py")
chunker = _load_module("chunker_mod", "3.chunking.py")
embedding_mod = _load_module("embedding_mod", "4.embedding.py")
vector_db_mod = _load_module("vector_db_mod", "5.vector_db.py")
retrieval_mod = _load_module("retrieval_mod", "6.retrieval.py")
rerank_mod = _load_module("rerank_mod", "7.reranking.py")

extract_pdf = pdf_extractor.extract_pdf
preprocess_document = preprocessor.preprocess_document
chunk_document = chunker.chunk_document
EmbeddingModel = embedding_mod.EmbeddingModel
build_vector_store = vector_db_mod.build_vector_store
retrieve_similar_chunks = retrieval_mod.retrieve_similar_chunks
rerank_chunks = rerank_mod.rerank_chunks


def build_rag_index(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Parse PDF, preprocess, chunk, embed, and store in vector DB."""
    raw_pages = extract_pdf(pdf_path)
    cleaned_text = preprocess_document(raw_pages)
    chunks = chunk_document(cleaned_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    model = EmbeddingModel()
    embeddings = model.embed_texts([chunk["text"] for chunk in chunks])

    vector_store = build_vector_store(chunks, embeddings)
    return chunks, embeddings, vector_store


def answer_question(pdf_path: str, question: str, top_k: int = 5) -> Dict[str, Any]:
    """Retrieve relevant chunks and rerank them for a question."""
    chunks, _, vector_store = build_rag_index(pdf_path)
    retrieved = retrieve_similar_chunks(question, vector_store, top_k=top_k)
    reranked = rerank_chunks(question, retrieved, top_n=3)

    context = "\n\n".join(item.get("text", "") for item in reranked)

    return {
        "question": question,
        "retrieved_chunks": reranked,
        "context": context,
    }


if __name__ == "__main__":
    result = answer_question("sample.pdf", "What is this document about?")
    print(result["question"])
    print("---")
    for idx, chunk in enumerate(result["retrieved_chunks"], 1):
        print(f"Chunk {idx}: {chunk.get('text', '')[:250]}...")
