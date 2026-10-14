"""Embedding backends. All return L2-normalised float32 vectors so dot product == cosine similarity."""

from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def _preload_msvc_runtime() -> None:
    """On Windows, onnxruntime needs a newer MSVC runtime than some machines have system-wide.
    The `msvc-runtime` wheel (a Windows-only requirement) ships one into the environment; load it
    first so onnxruntime binds to it instead of crashing on import."""
    if os.name != "nt":
        return
    dll = Path(sys.prefix) / "msvcp140.dll"
    if dll.exists():
        import ctypes

        ctypes.WinDLL(str(dll))


class FastEmbedEmbedder:
    """Local ONNX sentence-embedding model (default: BAAI/bge-small-en-v1.5, 384 dims). No API key."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        _preload_msvc_runtime()
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        # query_embed applies the model's retrieval instruction prefix (asymmetric search).
        return _normalise(np.stack(list(self._model.query_embed([text])))[0])


class HashingEmbedder:
    """Deterministic, dependency-free bag-of-words embedder. Lexical only: used for fast tests/offline."""

    def __init__(self, dims: int = 1024):
        self.name = f"hashing-{dims}"
        self.dims = dims

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dims, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for token in tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]:
            digest = hashlib.md5(token.encode()).digest()
            vec[int.from_bytes(digest[:4], "little") % self.dims] += 1.0 if digest[4] % 2 else -1.0
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._vector(text))


def build_embedder(backend: str, model_name: str, cache_dir: str | None = None) -> Embedder:
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name, cache_dir=cache_dir)
    if backend == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (expected 'fastembed' or 'hashing')")
