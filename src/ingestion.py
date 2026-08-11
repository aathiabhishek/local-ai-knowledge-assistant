from io import BytesIO
from typing import List, Dict

import pymupdf
from docx import Document as DocxDocument

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


def _clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def _chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    text = _clean_text(text)

    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer a natural boundary.
        if end < len(text):
            candidates = [
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind(" ", start, end),
            ]

            boundary = max(candidates)

            if boundary > start + int(chunk_size * 0.5):
                end = boundary + 1

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(end - overlap, start + 1)

    return chunks


def _load_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    documents = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = _clean_text(page.get_text())

            for chunk in _chunk_text(text):
                documents.append(
                    {
                        "text": chunk,
                        "source": filename,
                        "page": page_number,
                    }
                )

    return documents


def _load_docx(file_bytes: bytes, filename: str) -> List[Dict]:
    document = DocxDocument(BytesIO(file_bytes))

    text = "\n".join(
        paragraph.text
        for paragraph in document.paragraphs
        if paragraph.text.strip()
    )

    return [
        {
            "text": chunk,
            "source": filename,
            "page": None,
        }
        for chunk in _chunk_text(text)
    ]


def _load_text(file_bytes: bytes, filename: str) -> List[Dict]:
    text = file_bytes.decode("utf-8", errors="ignore")

    return [
        {
            "text": chunk,
            "source": filename,
            "page": None,
        }
        for chunk in _chunk_text(text)
    ]


def load_uploaded_files(uploaded_files) -> List[Dict]:
    all_documents = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        extension = filename.lower().split(".")[-1]

        if extension == "pdf":
            docs = _load_pdf(file_bytes, filename)

        elif extension == "docx":
            docs = _load_docx(file_bytes, filename)

        elif extension in {"txt", "md"}:
            docs = _load_text(file_bytes, filename)

        else:
            continue

        all_documents.extend(docs)

    return all_documents
