"""
📚 LEARN: Document Loaders — The First Step in RAG
====================================================
Before a RAG system can answer questions, it needs to READ your documents.
That's what this module does: it takes a file path or URL and returns plain text.

Why plain text? Because the next step (chunking) works on strings, and the
step after that (embedding) converts strings into vectors. Everything starts
with getting clean text out of your documents.

This module supports three formats:
  1. Plain text / Markdown files  (.txt, .md)
  2. PDF files                    (.pdf)
  3. Web pages                    (http:// or https://)

Each loader is a simple function: input → string of text.
"""
import os
# We import PyPDF2 for PDF parsing. PDFs are complex binary files — they contain fonts, images, layout data, and text all mixed together. 
# Writing a PDF parser from scratch would be a project on its own, so we use PyPDF2 for this one task.
from PyPDF2 import PdfReader

# requests lets us fetch web pages via HTTP. We could use Python's built-in urllib, but requests has a much cleaner API.
import requests

# BeautifulSoup parses HTML and extracts the visible text content,stripping away all the HTML tags, scripts, and styles. 
# Web pages are full of junk (navigation bars, footers, ads) — we want just the article text.
from bs4 import BeautifulSoup

from pdf2image import convert_from_path
import pytesseract # use for OCR - handle scanned PDFs

def load_text_file(file_path: str) -> str:
    """
    Load a plain text or Markdown file and return its contents as a string.

    This is the simplest loader — just read the file. Markdown is already
    mostly plain text, so it works great for RAG without any special parsing.

    Args:
        file_path: Path to a .txt or .md file

    Returns:
        The file's contents as a string
    """
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # We use encoding="utf-8" to handle special characters properly.
    # Without it, Python might use your system's default encoding, which can
    # cause errors on files with accented characters, emoji, etc.
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # .strip() removes leading/trailing whitespace. This is a small but important cleanup step — 
    # extra blank lines at the start/end of a document would waste space in our chunks later.
    return text.strip()

def load_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file and return it as a single string.

    PDFs store text in a complex binary format with fonts, coordinates, and
    rendering instructions. PyPDF2 does the heavy lifting of extracting the
    actual text content from each page.

    📚 LEARN — LIMITATION: PyPDF2 can only extract TEXT that is stored as
    text in the PDF. It CANNOT read text from images. This means:
      - ✅ Text typed in Word/Google Docs and exported as PDF → works great
      - ✅ Text + images mixed together → text is extracted, images are skipped
      - ❌ Scanned documents (photos of paper) → returns empty text
      - ❌ Text embedded inside diagrams/charts → not extracted

    To handle scanned PDFs, we would need OCR (Optical Character Recognition)
    via a library like pytesseract + Tesseract. 

    1. We prioritize to extract text via PyPDF2 for simple text pdf version 
    2. if it fails, we switch to OCR for extracting.(Scanned documents...)

    Args:
        file_path: Path to a .pdf file

    Returns:
        All text from all pages, concatenated with newlines between pages
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # 1. extract using PyPDF2
    reader = PdfReader(file_path)
    pages_text = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages_text.append(pages_text.strip())
    
    full_text = "\n\n".join(pages_text)

    if len(full_text.strip()) > 30:
        return full_text
    # if < 30 -> might be scanned doc, blank space, pdf with images(cuz PyPDF2 can't recognize)
    # if > 30 -> might be normal pdf

    # 2. if not text -> use OCR
    print("🔍 Can't not extract text, starting OCR recognition...")
    try:
        pages = convert_from_path(file_path, 300)
        ocr_text = []
        for page in pages:
            text = pytesseract.image_to_string(page, lang='eng')
            ocr_text.append(text.strip())
        return "\n\n".join(ocr_text)
    except Exception as e:
        print(f"OCR recognition failed: {e}")
        return full_text

def load_web_page(url: str) -> str:
    """
    Fetch a web page and extract its visible text content.

    Web pages are HTML — a mix of content, navigation, scripts, styles, and ads.
    We use BeautifulSoup to parse the HTML and extract just the readable text.

    Args:
        url: A URL starting with http:// or https://

    Returns:
        The visible text content of the page
    """
    # We set a User-Agent header so the web server knows we're a bot.
    # Some servers block requests without a User-Agent, returning 403 Forbidden.
    # Being transparent about what we are is good practice.
    headers = {
        "User-Agent": "RAG-From-Scratch/1.0 (Personal Project)"
    }

    response = requests.get(url, headers=headers, timeout=10) # try 10s
    response.raise_for_status() # Raise an error for 4xx/5xx status codes

    # BeautifulSoup parses the raw HTML string into a tree structure.
    # "html.parser" is Python's built-in HTML parser — no extra dependencies.
    soup = BeautifulSoup(response.text, "html.parser")

    # We remove <script> and <style> and other tags before extracting text.
    # Without this, we'd get JavaScript code and CSS rules mixed in with
    # the actual content — that would confuse the LLM later.
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    
    # get_text() walks the HTML tree and concatenates all visible text.
    # separator="\n" puts a newline between each element's text so paragraphs
    # don't merge into one giant line.
    text = soup.get_text(seperator="\n")

    # clean up: remove extra blank spaces in each line, and add '\n' line by line
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    
    return text

def load_document(source: str) -> dict:
    """
    Smart loader: auto-detect the source type and load accordingly.

    This is the main entry point for loading documents. It figures out whether
    the source is a URL, PDF, or text file, and calls the right loader.

    Args:
        source: A file path or URL

    Returns:
        A dict with keys:
          - "text": the extracted text content
          - "source": the original source path/URL
          - "type": "text", "pdf", or "web"
    """
    # This function is a simple "dispatcher" pattern. Instead of
    # making the caller figure out which loader to use, we detect the type
    # automatically. This makes the rest of the pipeline simpler — it just
    # calls load_document() and gets text back.
    if source.startswith("http://") or source.startswith("https://"):
        text = load_web_page(source)
        doc_type = "web"
    elif source.lower().endswith(".pdf"):
        text = load_pdf(source)
        doc_type = "pdf"
    else:
        text = load_text_file(souce)
        doc_type = "text"
    
    return {
        "text": text,
        "source": source,
        "type": doc_type,
    }

# The "if __name__ == '__main__'" block runs only when you execute
# this file directly (python loader.py), not when it's imported by other code.
# It's great for quick manual testing during development.
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2: # check whether pass parameters or not
        # if not, print messages below and then exit
        print("Usage: python -m rag.loader <file_path_or_url>")
        print("Examples:")
        print("  python -m rag.loader sample_data/sample.md")
        print("  python -m rag.loader sample_data/sample.pdf")
        print("  python -m rag.loader https://example.com")
        sys.exit(1)

    
    source = sys.argv[1] # path/web url you write in command line
    print(f"Loading: {source}")
    print("-" * 60)

    doc = load_document(source)
    print(f"Type: {doc['type']}")
    print(f"Length: {len(doc['text'])} characters")
    print("-" * 60)
    # Show first 500 chars as a preview
    preview = doc["text"][:500]
    print(preview)
    if len(doc["text"]) > 500:
        print(f"\n... ({len(doc['text']) - 500} more characters)")