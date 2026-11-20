# documents.py: resume PDF -> text -> overlapping chunks
import warnings

from PyPDF2 import PdfReader

from resume_analyzer.config import CHUNK_OVERLAP, CHUNK_SIZE

warnings.filterwarnings("ignore", module="PyPDF2")


def pdf_to_text(pdf) -> str:
    """Text of every page of a PDF (path or file-like object), pages separated by blank lines."""
    text = ""
    for page in PdfReader(pdf).pages:
        t = page.extract_text()
        if t:
            text += t + "\n\n"
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks for retrieval."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def pdf_to_chunks(pdf) -> list[str]:
    return chunk_text(pdf_to_text(pdf))
