"""
utils/parser.py
Handles text extraction from .txt, .pdf, and .docx resume files.
"""

import os
import re


def extract_text_from_txt(filepath: str) -> str:
    """Read plain text file."""
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from a PDF file using PyPDF2."""
    try:
        import PyPDF2
        text = []
        with open(filepath, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text.append(page.extract_text() or "")
        return "\n".join(text)
    except ImportError:
        raise ImportError("PyPDF2 is required for PDF parsing. Run: pip install PyPDF2")
    except Exception as e:
        raise RuntimeError(f"Failed to parse PDF '{filepath}': {e}")


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from a .docx Word file."""
    try:
        from docx import Document
        doc = Document(filepath)
        paragraphs = [para.text for para in doc.paragraphs]
        return "\n".join(paragraphs)
    except ImportError:
        raise ImportError("python-docx is required for DOCX parsing. Run: pip install python-docx")
    except Exception as e:
        raise RuntimeError(f"Failed to parse DOCX '{filepath}': {e}")


def parse_resume(filepath: str) -> str:
    """
    Auto-detect file type and extract raw text from a resume.

    Supported formats: .txt, .pdf, .docx
    Returns: raw text string
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".txt":
        return extract_text_from_txt(filepath)
    elif ext == ".pdf":
        return extract_text_from_pdf(filepath)
    elif ext == ".docx":
        return extract_text_from_docx(filepath)
    else:
        raise ValueError(f"Unsupported file format: '{ext}'. Supported: .txt, .pdf, .docx")


def clean_text(text: str) -> str:
    """
    Normalize whitespace, remove special characters, lowercase.
    Used before NLP processing.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\+\#\.]", " ", text)  # keep alphanumeric + C++, C#, etc.
    text = re.sub(r"\s+", " ", text).strip()
    return text
