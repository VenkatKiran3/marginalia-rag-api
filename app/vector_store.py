"""A small persistent vector store: a NumPy matrix for exact cosine search plus JSON chunk metadata.

Exact (brute-force) search is the right call at this scale — tens of thousands of chunks search in
milliseconds, results are deterministic, and there is no extra service to run. The class exposes the
same add / delete / search surface a hosted store (Qdrant, pgvector, Chroma) would, so it can be
swapped without touching the RAG pipeline.
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from pathlib import Path

import numpy as np

from app.ingestion import Chunk


class VectorStore:
    def __init__(self, path: Path | None = None):
        self.path = path
        self._lock = threading.RLock()
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._embedder_name: str | None = None
        if path:
            self._load()

    # ---------------------------------------------------------------- mutations
    def add(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self._lock:
            if self._embedder_name and self._embedder_name != embedder_name:
                raise ValueError(
                    f"Index was built with '{self._embedder_name}'; re-index to use '{embedder_name}'."
                )
            self._embedder_name = embedder_name
            vectors = np.asarray(vectors, dtype=np.float32)
            self._vectors = vectors if self._vectors is None else np.vstack([self._vectors, vectors])
            self._chunks.extend(chunks)
            self._save()

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            keep = [i for i, c in enumerate(self._chunks) if c.doc_id != doc_id]
            removed = len(self._chunks) - len(keep)
            if removed:
                self._chunks = [self._chunks[i] for i in keep]
                self._vectors = self._vectors[keep] if keep else None
                if not keep:
                    self._embedder_name = None
                self._save()
            return removed

    # ---------------------------------------------------------------- queries
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        with self._lock:
            if self._vectors is None or not self._chunks:
                return []
            scores = self._vectors @ np.asarray(query_vector, dtype=np.float32)
            k = min(k, len(scores))
            top = np.argpartition(-scores, k - 1)[:k]
            top = top[np.argsort(-scores[top])]
            return [(self._chunks[i], float(scores[i])) for i in top]

    def documents(self) -> list[dict]:
        with self._lock:
            docs: dict[str, dict] = {}
            for chunk in self._chunks:
                entry = docs.setdefault(
                    chunk.doc_id,
                    {"id": chunk.doc_id, "name": chunk.document, "chunks": 0, "pages": set()},
                )
                entry["chunks"] += 1
                if chunk.page:
                    entry["pages"].add(chunk.page)
            return [{**d, "pages": len(d["pages"]) or None} for d in docs.values()]

    def has_document(self, doc_id: str) -> bool:
        with self._lock:
            return any(c.doc_id == doc_id for c in self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    # ---------------------------------------------------------------- persistence
    def _save(self) -> None:
        if not self.path:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        meta = {"embedder": self._embedder_name, "chunks": [asdict(c) for c in self._chunks]}
        tmp = self.path / "chunks.json.tmp"
        tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        vectors = self._vectors if self._vectors is not None else np.zeros((0, 0), np.float32)
        np.save(self.path / "vectors.npy", vectors)
        tmp.replace(self.path / "chunks.json")

    def _load(self) -> None:
        meta_file, vec_file = self.path / "chunks.json", self.path / "vectors.npy"
        if not (meta_file.exists() and vec_file.exists()):
            return
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        self._chunks = [Chunk(**c) for c in meta["chunks"]]
        self._embedder_name = meta.get("embedder")
        vectors = np.load(vec_file)
        self._vectors = vectors if len(self._chunks) else None
