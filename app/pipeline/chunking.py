"""Structure-aware markdown chunking: split on headers, pack paragraphs.

Chunk size is word-based (~600 words ≈ 800 tokens) with paragraph-boundary
overlap. Output order is deterministic: doc order then section order.
"""
import hashlib
import re

CHUNK_WORDS = 600
OVERLAP_WORDS = 80

_HEADER = re.compile(r"^(#{1,6})\s+(.*)$")


def _sections(markdown: str) -> list[tuple[list[str], list[str]]]:
    """Split into (heading_path, paragraphs) sections."""
    sections: list[tuple[list[str], list[str]]] = []
    heading_stack: list[str] = []
    buf: list[str] = []
    for line in markdown.splitlines():
        m = _HEADER.match(line)
        if m:
            if buf:
                sections.append((list(heading_stack), buf))
                buf = []
            level = len(m.group(1))
            heading_stack = heading_stack[: level - 1] + [m.group(2).strip()]
        else:
            buf.append(line)
    if buf:
        sections.append((list(heading_stack), buf))
    return sections


def chunk_document(doc_id: str, url: str, title: str, markdown: str) -> list[dict]:
    chunks: list[dict] = []
    idx = 0
    for heading_path, lines in _sections(markdown):
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", "\n".join(lines)) if p.strip()]
        current: list[str] = []
        current_words = 0
        for para in paragraphs:
            w = len(para.split())
            if current and current_words + w > CHUNK_WORDS:
                chunks.append(_make(doc_id, url, title, heading_path, current, idx))
                idx += 1
                tail = " ".join(current).split()[-OVERLAP_WORDS:]
                current = [" ".join(tail), para] if tail else [para]
                current_words = len(" ".join(current).split())
            else:
                current.append(para)
                current_words += w
        if current:
            chunks.append(_make(doc_id, url, title, heading_path, current, idx))
            idx += 1
    return chunks


def _make(doc_id: str, url: str, title: str, heading_path: list[str],
          paragraphs: list[str], idx: int) -> dict:
    text = "\n\n".join(paragraphs)
    return {
        "chunk_id": hashlib.sha256(f"{doc_id}:{idx}:{text}".encode()).hexdigest()[:16],
        "doc_id": doc_id,
        "source_url": url,
        "title": title,
        "section": " > ".join(heading_path) if heading_path else None,
        "text": text,
    }
