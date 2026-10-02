from __future__ import annotations
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from bs4 import BeautifulSoup
from ..models import Document, DocumentFormat
import pypdf

def _format_for(path: Path, override: DocumentFormat | None) -> DocumentFormat:
    if override is not None:
        return override
    suffix = path.suffix.lower()
    if suffix in {".txt"}:
        return DocumentFormat.TXT
    if suffix in {".md", ".markdown"}:
        return DocumentFormat.MARKDOWN
    if suffix in {".html", ".htm"}:
        return DocumentFormat.HTML
    if suffix in {".pdf"}:
        return DocumentFormat.PDF
    if suffix in {".docx", ".doc"}:
        return DocumentFormat.DOCX
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
    

def read_pdf(path: Path) -> str:
    reader = pypdf.PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)

def read_html(raw_html: str) -> tuple[str, str]:
    soup = BeautifulSoup(raw_html, "html.parser")
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""
    body = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in body.splitlines()] 
    plain = "\n".join(ln for ln in lines if ln)
    return title, plain

def title_from_markdown(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return ""

def load_document(path: Path, format_override: DocumentFormat | None = None) -> Document:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    format = _format_for(path, format_override)
    if format is DocumentFormat.PDF:
        text = read_pdf(path)
        title = path.stem
    elif format is DocumentFormat.HTML:
        raw = path.read_text(encoding="utf-8", errors="replace")
        title, text = read_html(raw)
    elif format is DocumentFormat.MARKDOWN:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = title_from_markdown(text) or path.stem
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        title = path.stem

    text = text.strip()

    if not text:
        raise ValueError(f"Document is empty: {path}")
    return Document(
        source=str(path),
        format=format,
        title=title,
        text=text,
        metadata={"path": str(path), "size_chars": str(len(text)) },
    )

def load_path(path: Path, *, extensions: Iterable[str] = (".txt", ".md", ".markdown", ".html", ".htm", ".pdf", ".docx")) -> Iterator[Document]:
    if path.is_file():
        
        yield load_document(path)
        return
    exts = {e.lower() for e in extensions}
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        if child.suffix.lower() not in exts:
            continue
        try:
            yield load_document(child)
        except (ValueError, FileNotFoundError):
            continue
        
