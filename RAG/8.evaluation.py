"""
Evaluation and hallucination-reduction utilities for RAG.
"""

from typing import Any, Dict, List, Sequence


def retrieval_precision(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    if not retrieved:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in retrieved if item in relevant_set)
    return hits / len(retrieved)


def retrieval_recall(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in relevant if item in set(retrieved))
    return hits / len(relevant)


def evaluate_retrieval(retrieved_ids: Sequence[str], relevant_ids: Sequence[str]) -> Dict[str, float]:
    return {
        "precision": retrieval_precision(retrieved_ids, relevant_ids),
        "recall": retrieval_recall(retrieved_ids, relevant_ids),
    }


def hallucination_risk_score(answer: str, context_chunks: Sequence[str]) -> float:
    """Very lightweight heuristic: if answer contains terms not found in the contexts, risk is higher."""
    context_text = " ".join(context_chunks).lower()
    answer_lower = answer.lower()

    if not context_chunks:
        return 1.0

    # Count uncommon terms in the answer compared to context
    answer_tokens = set(answer_lower.split())
    context_tokens = set(context_text.split())
    missing = sum(1 for token in answer_tokens if token not in context_tokens and len(token) > 3)
    return min(1.0, missing / max(1, len(answer_tokens)))


def reduce_hallucination(answer: str, context_chunks: Sequence[str]) -> str:
    """Simple answer guardrail: encourage the model to ground the answer in retrieved context."""
    if not context_chunks:
        return "I could not find relevant context to answer this question confidently."

    risk = hallucination_risk_score(answer, context_chunks)
    if risk > 0.5:
        return (
            "Answer based only on the retrieved documents. "
            "If the context does not contain enough information, say that the answer is not available."
        )
    return answer
