import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf
from docx import Document as DocxDocument
from docx.table import Table
from docx.text.paragraph import Paragraph

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


_WHITESPACE = re.compile(r"\s")

# Preferred split points, strongest first.
_BOUNDARIES = ("\n\n", "\n", ". ", " ")


def _clean_text(text: str) -> str:
    """Normalize whitespace but keep indentation and paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_boundary(text: str, start: int, end: int, min_pos: int) -> int:
    """Pick the strongest natural boundary in the second half of the window."""
    for sep in _BOUNDARIES:
        pos = text.rfind(sep, start, end)
        if pos >= min_pos:
            return pos + len(sep)
    return end


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
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)

        if end < n:
            end = _find_boundary(
                text, start, end, start + int(chunk_size * 0.5)
            )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        next_start = max(end - overlap, start + 1)

        # Don't begin the next chunk in the middle of a word.
        if not text[next_start - 1].isspace():
            match = _WHITESPACE.search(text, next_start, end)
            if match:
                next_start = match.end()

        start = next_start

    return chunks


def _make_docs(
    chunks: List[str],
    filename: str,
    page: Optional[int] = None,
) -> List[Dict]:
    return [
        {"text": chunk, "source": filename, "page": page}
        for chunk in chunks
    ]


def _load_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    documents = []

    with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text", sort=True)
            documents.extend(
                _make_docs(_chunk_text(text), filename, page_number)
            )

    return documents


def _iter_docx_blocks(document):
    """Yield paragraphs and tables in document order."""
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]

        if tag == "p":
            yield Paragraph(child, document)
        elif tag == "tbl":
            yield Table(child, document)


def _table_to_text(table: Table) -> str:
    rows = []

    for row in table.rows:
        cells = []

        for cell in row.cells:
            value = " ".join(cell.text.split())

            # Merged cells repeat the same text; skip consecutive duplicates.
            if value and (not cells or cells[-1] != value):
                cells.append(value)

        if cells:
            rows.append(" | ".join(cells))

    return "\n".join(rows)


def _load_docx(file_bytes: bytes, filename: str) -> List[Dict]:
    document = DocxDocument(BytesIO(file_bytes))

    blocks = []

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = block.text.strip()
        else:
            text = _table_to_text(block)

        if text:
            blocks.append(text)

    # Blank line between blocks so chunking prefers paragraph boundaries.
    return _make_docs(_chunk_text("\n\n".join(blocks)), filename)


def _load_text(file_bytes: bytes, filename: str) -> List[Dict]:
    text = file_bytes.decode("utf-8-sig", errors="replace")
    return _make_docs(_chunk_text(text), filename)


def load_uploaded_files(
    uploaded_files,
    warnings: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Load and chunk uploaded files.

    If a list is passed as `warnings`, messages about unsupported,
    unreadable or empty files are appended to it so the UI can show them.
    """

    def warn(message: str) -> None:
        if warnings is not None:
            warnings.append(message)

    all_documents = []

    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.getvalue()
        filename = uploaded_file.name
        extension = Path(filename).suffix.lower().lstrip(".")

        try:
            if extension == "pdf":
                docs = _load_pdf(file_bytes, filename)

            elif extension == "docx":
                docs = _load_docx(file_bytes, filename)

            elif extension in {"txt", "md"}:
                docs = _load_text(file_bytes, filename)

            else:
                warn(f"{filename}: unsupported file type, skipped.")
                continue

        except Exception as exc:
            warn(f"{filename}: could not be read ({exc}).")
            continue

        if not docs:
            if extension == "pdf":
                warn(
                    f"{filename}: no text found. It may be a scanned PDF "
                    "that needs OCR."
                )
            else:
                warn(f"{filename}: no text found.")
            continue

        all_documents.extend(docs)

    return all_documents