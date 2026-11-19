"""
step 3 is chunking:
input cleaned PDF text and split it into smaller chunks
so each chunk can later be embedded and stored in a vector database.
"""

from typing import List, Dict, Any

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore
except ImportError:  # pragma: no cover
    RecursiveCharacterTextSplitter = None


def split_text_with_overlap(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split a text string into chunks with overlap.
    If LangChain is available, use its recursive splitter.
    Otherwise, use a simple fallback chunker.
    """
    if not text or not text.strip():
        return []

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    # Simple fallback chunker when LangChain is not installed
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start += chunk_size - chunk_overlap

    return chunks


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split a full cleaned document into text chunks.
    """
    return split_text_with_overlap(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_pages(cleaned_pages: List[Dict[str, Any]], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split cleaned pages into chunk dictionaries with metadata.

    Example output:
    [{
        "chunk_id": "page_1_chunk_1",
        "page": 1,
        "text": "...",
        "metadata": {"page": 1}
    }]
    """
    chunks = []

    for page in cleaned_pages:
        page_number = page.get("page", 1)
        page_text = page.get("text", "")
        page_chunks = split_text_with_overlap(
            page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_index, chunk in enumerate(page_chunks, start=1):
            chunks.append({
                "chunk_id": f"page_{page_number}_chunk_{chunk_index}",
                "page": page_number,
                "text": chunk.strip(),
                "metadata": {"page": page_number}
            })

    return chunks


def chunk_document(document_text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict[str, Any]]:
    """
    Split a full document string into chunks with a sequential chunk_id.
    """
    text_chunks = split_text_with_overlap(document_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    return [
        {
            "chunk_id": f"chunk_{index}",
            "text": chunk.strip(),
            "metadata": {}
        }
        for index, chunk in enumerate(text_chunks, start=1)
        if chunk.strip()
    ]


# Example usage:
# extracted_pages = [
#     {"page": 1, "text": "..."},
#     {"page": 2, "text": "..."},
# ]
# chunks = chunk_pages(extracted_pages)
# print(f"Created {len(chunks)} chunks.")