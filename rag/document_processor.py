"""Turn uploaded PDF files into text chunks with filename, page and chunk ID metadata."""

import hashlib
import os
import tempfile
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
)


def file_hash(data: bytes) -> str:
    """Fingerprint of the file contents, used to detect PDFs that are already indexed."""
    return hashlib.sha256(data).hexdigest()


def load_pdf_pages(data: bytes) -> List[Document]:
    """Extract text from a PDF, one Document per page."""
    # PyPDFLoader needs a file path. delete=False lets us close the temp file
    # before reading it, which Windows requires.
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return PyPDFLoader(tmp_path).load()
    finally:
        os.remove(tmp_path)


def split_pdf(file_name: str, data: bytes) -> List[Document]:
    """Split a PDF into chunks. Each chunk remembers its file, page and a stable ID."""
    doc_hash = file_hash(data)
    chunks = []

    # Split page by page so every chunk has an exact page number.
    for page in load_pdf_pages(data):
        page_number = page.metadata.get("page", 0) + 1  # PyPDFLoader counts from 0

        for i, text in enumerate(_splitter.split_text(page.page_content)):
            if not text.strip():
                continue
            # Same file -> same IDs, so re-processing a PDF never creates duplicates.
            chunk_id = f"{doc_hash[:16]}-p{page_number}-c{i}"
            chunks.append(
                Document(
                    id=chunk_id,
                    page_content=text,
                    metadata={
                        "source": file_name,
                        "page": page_number,
                        "chunk_id": chunk_id,
                        "file_hash": doc_hash,
                    },
                )
            )

    return chunks
