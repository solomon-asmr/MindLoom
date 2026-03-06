import os
import csv
import fitz
from docx import Document


def load_pdf(file_path):
    """Extract text from a PDF file."""
    doc = fitz.open(file_path)
    
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    
    doc.close()
    return text.strip()


def load_txt(file_path):
    """Read a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def load_docx(file_path):
    """Extract text from a Word document."""
    doc = Document(file_path)
    
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    
    return text.strip()


def load_csv(file_path):
    """Extract text from a CSV file."""
    text = ""
    
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            text += " ".join(row) + "\n"
    
    return text.strip()


LOADERS = {
    ".pdf": load_pdf,
    ".txt": load_txt,
    ".docx": load_docx,
    ".csv": load_csv,
}


def load_document(file_path):
    """Auto-detect file type and extract text."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext not in LOADERS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {list(LOADERS.keys())}")
    
    text = LOADERS[ext](file_path)
    
    if not text:
        raise ValueError(f"No text could be extracted from {file_path}")
    
    return text


def get_supported_extensions():
    """Return list of supported file extensions."""
    return list(LOADERS.keys())