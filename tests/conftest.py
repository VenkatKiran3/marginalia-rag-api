from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.embeddings import HashingEmbedder
from app.llm import Context, GeneratedAnswer, Quote
from app.main import build_service, create_app
from app.rag import RAGService
from app.vector_store import VectorStore

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


class FakeLLM:
    """Scriptable stand-in for Claude. By default it 'answers' from source 1, quoting its first sentence."""

    model = "fake-llm"

    def __init__(self, response: GeneratedAnswer | None = None):
        self.response = response
        self.calls: list[tuple[str, list[Context]]] = []

    def generate(self, question: str, contexts: list[Context]) -> GeneratedAnswer:
        self.calls.append((question, contexts))
        if self.response is not None:
            return self.response
        first_sentence = contexts[0].text.split(". ")[0]
        return GeneratedAnswer("answered", f"{first_sentence} [1].", [Quote(1, first_sentence)])


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(_env_file=None, anthropic_api_key=None, embedding_backend="hashing",
                    data_dir=tmp_path, min_relevance=0.05, seed_dir=None)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def rag(settings, fake_llm) -> RAGService:
    return build_service(settings, embedder=HashingEmbedder(), llm=fake_llm, store=VectorStore())


@pytest.fixture
def seeded_rag(rag) -> RAGService:
    for path in sorted(SAMPLE_DIR.iterdir()):
        rag.ingest(path.name, path.read_bytes())
    return rag


@pytest.fixture
def client(settings, rag) -> TestClient:
    with TestClient(create_app(settings, rag=rag)) as test_client:
        yield test_client
