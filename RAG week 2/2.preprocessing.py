"""
step 2 is preprocessing:
input the data extracted from the pdf and clean it up, 
remove any unwanted characters,
 and prepare it for chunking.
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """
    Clean raw extracted text from PDF pages.
    - normalize line endings
    - remove unwanted spacing
    - remove repeated whitespace
    - fix broken hyphenation in PDF extraction
    """
    if not text:
        return ""

    # Normalize separators
    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove hyphenation broken across lines, e.g. "infor-\nmation" -> "information"
    text = text.replace("-\n", "")
    text = text.replace("- \n", "")

    # Remove weird control characters
    text = re.sub(r"[\x00-\x1F\x7F]+", " ", text)

    # Reduce repeated blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    # Normalize spaces within each line
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)

    cleaned = "\n".join(lines)

    # Fix spacing around punctuation
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])(?=\S)", r"\1 ", cleaned)

    return cleaned.strip()


def preprocess_pages(extracted_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Input: list from extract_pdf() such as:
    [{"page": 1, "text": "..."}, {"page": 2, "text": "..."}]

    Output: cleaned list of pages ready for chunking.
    """
    cleaned_pages = []

    for page in extracted_pages:
        page_number = page.get("page", 1)
        raw_text = page.get("text", "")
        cleaned_text = clean_text(raw_text)

        if cleaned_text:
            cleaned_pages.append({
                "page": page_number,
                "text": cleaned_text,
            })

    return cleaned_pages


def preprocess_document(extracted_pages: List[Dict[str, Any]]) -> str:
    """
    Combine all cleaned pages into one text string for chunking.
    """
    cleaned_pages = preprocess_pages(extracted_pages)
    return "\n\n".join(page["text"] for page in cleaned_pages)


# Example usage:
# from 1.pdf_extractor import extract_pdf
# raw_pages = extract_pdf("sample.pdf")
# cleaned_pages = preprocess_pages(raw_pages) // i think a loop is needed here 
# full_text = preprocess_document(raw_pages)
