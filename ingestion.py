# This file handles reading the PDF and splitting it into manageable, overlapping chunks while preserving the page numbers.

from pypdf import PdfReader

def extract_text_from_pdf(pdf_file):
    """Extracts text from a uploaded PDF file page by page."""
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            # i + 1 because page numbers usually start at 1
            pages.append({"text": text, "page_number": i + 1})
    return pages

def chunk_text(pages, chunk_size=400, overlap=50):
    """
    Splits page text into overlapping chunks of words.
    Returns a list of dictionaries containing the chunk text and source page.
    """
    chunks = []
    for page in pages:
        text = page["text"]
        words = text.split()
        i = 0
        
        while i < len(words):
            # Grab a slice of words
            chunk_words = words[i:i + chunk_size]
            chunk_text_str = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text_str,
                "page_number": page["page_number"]
            })
            
            # Advance by chunk_size minus overlap
            i += (chunk_size - overlap)
            
    return chunks