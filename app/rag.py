"""The retrieval-augmented answering pipeline and its reliability guardrails.

Answering a question passes four gates, each of which can only make the answer *more* conservative:

1. Relevance gate   – if even the best chunk scores below `min_relevance`, the LLM is never called.
2. Grounded prompt  – the model may only use the numbered excerpts and must cite them.
3. Citation check   – citations to sources the model was never shown are discarded; an answer
                      left with no valid citation is downgraded to "unanswerable".
4. Quote check      – each evidence quote must appear (near-)verbatim in its source chunk; if not,
                      the excerpt shown is the retrieved text itself, never the model's paraphrase.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from app.embeddings import Embedder
from app.ingestion import Chunk, chunk_document, document_id
from app.llm import AnswerGenerator, Context, GeneratedAnswer
from app.vector_store import VectorStore

UNAVAILABLE_MESSAGE = "The documents don't contain information to answer this question."
NO_DOCUMENTS_MESSAGE = "No documents have been added yet. Upload documents before asking questions."


class LLMNotConfiguredError(RuntimeError):
    pass


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Evidence:
    source: int
    document: str
    page: int | None
    heading: str | None
    excerpt: str
    score: float
    quote_verified: bool


@dataclass
class Answer:
    question: str
    status: str  # answered | partial | unanswerable
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    missing: str = ""
    top_score: float | None = None
    latency_ms: int = 0


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[‘’]", "'", re.sub(r"[“”]", '"', text))).strip().lower()


def verify_quote(quote: str, source_text: str, threshold: float = 0.9) -> bool:
    """True if `quote` appears in `source_text`, tolerating whitespace/quote-style/PDF-extraction noise."""
    q, s = _normalise(quote), _normalise(source_text)
    if not q:
        return False
    if q in s:
        return True
    match = SequenceMatcher(None, s, q, autojunk=False).find_longest_match(0, len(s), 0, len(q))
    if match.size / len(q) >= threshold:
        return True
    # Fall back to a fuzzy comparison against the best-aligned window of the source.
    start = max(0, match.a - match.b)
    window = s[start : start + len(q) + 20]
    return SequenceMatcher(None, window, q, autojunk=False).ratio() >= threshold


_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def strip_invalid_citations(answer: str, valid: set[int]) -> tuple[str, set[int]]:
    """Remove citation markers pointing at sources that were never provided; return those kept."""
    used: set[int] = set()

    def keep(match: re.Match) -> str:
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1)) if int(n) in valid]
        used.update(numbers)
        return f"[{', '.join(map(str, numbers))}]" if numbers else ""

    cleaned = _CITATION.sub(keep, answer)
    return re.sub(r"\s+([.,;:])", r"\1", cleaned).strip(), used


def _excerpt(text: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[: text.rfind(" ", 0, limit)] + " …"


class RAGService:
    def __init__(self, store: VectorStore, embedder: Embedder, llm: AnswerGenerator | None,
                 chunk_size: int = 700, chunk_overlap: int = 120, top_k: int = 8,
                 min_relevance: float = 0.5):
        self.store = store
        self.embedder = embedder
        self.llm = llm
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.min_relevance = min_relevance

    # ------------------------------------------------------------------ ingestion
    def ingest(self, filename: str, data: bytes) -> dict:
        """Parse, chunk, embed and index one file. Re-uploading a filename replaces the old version."""
        chunks = chunk_document(filename, data, self.chunk_size, self.chunk_overlap)
        vectors = self.embedder.embed_documents([self._embedding_text(c) for c in chunks])
        replaced = self.store.delete_document(document_id(filename)) > 0
        self.store.add(chunks, vectors, self.embedder.name)
        pages = {c.page for c in chunks if c.page}
        return {"id": chunks[0].doc_id, "name": filename, "chunks": len(chunks),
                "pages": len(pages) or None, "replaced": replaced}

    @staticmethod
    def _embedding_text(chunk: Chunk) -> str:
        # Prefixing the section heading gives short chunks the context their heading provides.
        return f"{chunk.heading}\n{chunk.text}" if chunk.heading else chunk.text

    # ------------------------------------------------------------------ retrieval
    def retrieve(self, query: str, top_k: int | None = None) -> list[Hit]:
        if not query.strip():
            return []
        results = self.store.search(self.embedder.embed_query(query), top_k or self.top_k)
        return [Hit(chunk, score) for chunk, score in results]

    # ------------------------------------------------------------------ answering
    def answer(self, question: str, top_k: int | None = None) -> Answer:
        started = time.perf_counter()

        def done(result: Answer) -> Answer:
            result.latency_ms = int((time.perf_counter() - started) * 1000)
            return result

        if len(self.store) == 0:
            return done(Answer(question, "unanswerable", NO_DOCUMENTS_MESSAGE, missing="No documents indexed."))
        if self.llm is None:
            raise LLMNotConfiguredError("No LLM configured. Add your Gemini API key in Settings (gear icon) to enable answers.")

        hits = self.retrieve(question, top_k)
        top_score = hits[0].score if hits else None
        if not hits or hits[0].score < self.min_relevance:  # Gate 1
            return done(Answer(question, "unanswerable", UNAVAILABLE_MESSAGE, top_score=top_score,
                               missing="No passage in the documents is relevant to this question."))

        contexts = [
            Context(i, h.chunk.document, h.chunk.text, h.chunk.page, h.chunk.heading)
            for i, h in enumerate(hits, start=1)
        ]
        generated = self.llm.generate(question, contexts)  # Gate 2
        result = self._apply_guardrails(question, generated, hits)
        result.top_score = top_score
        return done(result)

    def _apply_guardrails(self, question: str, generated: GeneratedAnswer, hits: list[Hit]) -> Answer:
        def unavailable(missing: str) -> Answer:
            return Answer(question, "unanswerable", UNAVAILABLE_MESSAGE, missing=missing)

        if generated.status == "unanswerable" or not generated.answer.strip():
            return unavailable(generated.missing or "The retrieved passages do not answer the question.")

        valid = set(range(1, len(hits) + 1))
        text, cited = strip_invalid_citations(generated.answer, valid)  # Gate 3
        cited |= {q.source for q in generated.evidence if q.source in valid}
        if not cited:
            return unavailable("The model produced an answer without supporting citations, so it was withheld.")

        evidence: list[Evidence] = []
        for number in sorted(cited):  # Gate 4
            hit = hits[number - 1]
            quotes = [q.quote for q in generated.evidence if q.source == number]
            verified = next((q for q in quotes if verify_quote(q, hit.chunk.text)), None)
            evidence.append(Evidence(
                source=number,
                document=hit.chunk.document,
                page=hit.chunk.page,
                heading=hit.chunk.heading,
                excerpt=verified.strip() if verified else _excerpt(hit.chunk.text),
                score=round(hit.score, 4),
                quote_verified=verified is not None,
            ))
        return Answer(question, generated.status, text, evidence, generated.missing)
