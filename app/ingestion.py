"""Turn PDF / TXT / Markdown files into overlapping, metadata-rich text chunks."""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class IngestionError(ValueError):
    """Raised when a document cannot be turned into searchable text."""


@dataclass
class Section:
    """A contiguous span of a document: a PDF page, a Markdown section, or a whole text file."""

    text: str
    page: int | None = None
    heading: str | None = None


@dataclass
class Chunk:
    id: str
    doc_id: str
    document: str
    text: str
    index: int
    page: int | None = None
    heading: str | None = None
    metadata: dict = field(default_factory=dict)


def extension_of(filename: str) -> str:
    dot = filename.rfind(".")
    return filename[dot:].lower() if dot != -1 else ""


def parse_document(filename: str, data: bytes) -> list[Section]:
    ext = extension_of(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        raise IngestionError(
            f"Unsupported file type '{ext or 'none'}'. Upload a PDF, TXT or Markdown file."
        )
    if ext == ".pdf":
        sections = _parse_pdf(data)
    else:
        text = _decode(data)
        sections = _parse_markdown(text) if ext in {".md", ".markdown"} else [Section(text)]

    sections = [s for s in sections if s.text.strip()]
    if not sections:
        raise IngestionError(f"'{filename}' contains no extractable text.")
    return sections


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_pdf(data: bytes) -> list[Section]:
    try:
        reader = PdfReader(io.BytesIO(data))
        return [
            Section(text=page.extract_text() or "", page=number)
            for number, page in enumerate(reader.pages, start=1)
        ]
    except Exception as exc:  # pypdf raises a variety of errors on malformed files
        raise IngestionError(f"Could not read PDF: {exc}") from exc


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")


def _parse_markdown(text: str) -> list[Section]:
    """Split Markdown at headings so every chunk knows which section it came from."""
    sections: list[Section] = []
    trail: list[str] = []  # heading breadcrumb, e.g. ["Handbook", "Leave"]
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            sections.append(Section("\n".join(buffer), heading=" › ".join(trail) or None))
            buffer.clear()

    in_code = False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_code = not in_code
        match = None if in_code else _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            trail[:] = trail[: level - 1] + [match.group(2).strip()]
        else:
            buffer.append(line)
    flush()
    return sections


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


def _units(text: str, max_len: int) -> list[str]:
    """Break text into paragraphs, then sentences, then words, so no unit exceeds max_len."""
    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = re.sub(r"[ \t]*\n[ \t]*", " ", paragraph).strip()
        paragraph = re.sub(r"\s{2,}", " ", paragraph)
        if not paragraph:
            continue
        for sentence in _SENTENCE_END.split(paragraph):
            while len(sentence) > max_len:
                cut = sentence.rfind(" ", 0, max_len)
                cut = cut if cut > 0 else max_len
                units.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                units.append(sentence)
        units.append("\n")  # paragraph boundary marker
    return units


def split_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    """Greedily pack sentences into chunks of ~chunk_size chars, carrying ~overlap chars forward."""
    if overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    current: list[str] = []

    def render(parts: list[str]) -> str:
        out = ""
        for part in parts:
            if part == "\n":
                out = out.rstrip() + "\n\n"
            else:
                out += ("" if not out or out.endswith("\n") else " ") + part
        return out.strip()

    def length(parts: list[str]) -> int:
        return len(render(parts))

    for unit in _units(text, chunk_size):
        if unit != "\n" and current and length(current + [unit]) > chunk_size:
            chunks.append(render(current))
            # Seed the next chunk with trailing sentences that fit inside the overlap budget.
            carry: list[str] = []
            for part in reversed([p for p in current if p != "\n"]):
                if length([part] + carry) > overlap:
                    break
                carry.insert(0, part)
            current = carry
        current.append(unit)

    tail = render(current)
    if tail and (not chunks or tail != chunks[-1]):
        chunks.append(tail)
    return [c for c in chunks if c]


def document_id(filename: str) -> str:
    return hashlib.sha1(filename.lower().encode()).hexdigest()[:12]


def chunk_document(filename: str, data: bytes, chunk_size: int, overlap: int) -> list[Chunk]:
    doc_id = document_id(filename)
    chunks: list[Chunk] = []
    for section in parse_document(filename, data):
        for piece in split_text(section.text, chunk_size, overlap):
            chunks.append(
                Chunk(
                    id=f"{doc_id}-{len(chunks)}",
                    doc_id=doc_id,
                    document=filename,
                    text=piece,
                    index=len(chunks),
                    page=section.page,
                    heading=section.heading,
                )
            )
    if not chunks:
        raise IngestionError(f"'{filename}' contains no extractable text.")
    return chunks
