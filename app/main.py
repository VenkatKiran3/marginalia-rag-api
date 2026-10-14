"""Application factory: wires settings, embedder, vector store and LLM into a FastAPI app."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api import router
from app.config import Settings, get_settings
from app.embeddings import Embedder, build_embedder
from app.ingestion import SUPPORTED_EXTENSIONS, IngestionError, extension_of
from app.llm import AnswerGenerator, ClaudeAnswerGenerator, GeminiAnswerGenerator
from app.rag import RAGService
from app.vector_store import VectorStore

log = logging.getLogger("marginalia")
STATIC_DIR = Path(__file__).parent / "static"


def build_service(settings: Settings, embedder: Embedder | None = None,
                  llm: AnswerGenerator | None = None, store: VectorStore | None = None) -> RAGService:
    embedder = embedder or build_embedder(
        settings.embedding_backend, settings.embedding_model,
        str(settings.embedding_cache_dir or settings.data_dir / "models"),
    )
    if llm is None and settings.llm_provider == "gemini" and settings.gemini_api_key:
        llm = GeminiAnswerGenerator(settings.gemini_api_key, settings.gemini_model)
    elif llm is None and settings.llm_provider == "anthropic" and settings.anthropic_api_key:
        llm = ClaudeAnswerGenerator(settings.anthropic_api_key, settings.llm_model, settings.llm_effort,
                                    settings.llm_max_tokens, settings.llm_server_fallbacks)
    return RAGService(
        store=store if store is not None else VectorStore(settings.data_dir / "index"),
        embedder=embedder,
        llm=llm,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
        min_relevance=settings.min_relevance,
    )


def seed(rag: RAGService, folder: Path) -> None:
    """Index every supported file in `folder` when the store is empty (first boot)."""
    if len(rag.store) or not folder.is_dir():
        return
    for path in sorted(folder.iterdir()):
        if extension_of(path.name) in SUPPORTED_EXTENSIONS:
            try:
                info = rag.ingest(path.name, path.read_bytes())
                log.info("Seeded %s (%d chunks)", path.name, info["chunks"])
            except IngestionError as exc:
                log.warning("Skipped %s: %s", path.name, exc)


def create_app(settings: Settings | None = None, rag: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.rag is None:
            app.state.rag = build_service(settings)
        if settings.seed_dir:
            seed(app.state.rag, settings.seed_dir)
        if app.state.rag.llm is None:
            log.warning("No LLM API key set: add one in the UI's Settings (or GEMINI_API_KEY). /ask returns 503 until then.")
        yield

    app = FastAPI(
        title="Marginalia",
        description="Grounded question answering over your documents, with cited evidence.",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.rag = rag
    app.include_router(router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
app = create_app()
