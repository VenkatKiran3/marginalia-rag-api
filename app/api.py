"""HTTP routes. Thin layer: validation and status codes here, behaviour in RAGService."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.ingestion import IngestionError
from app.llm import GeminiAnswerGenerator, LLMError
from app.rag import LLMNotConfiguredError, RAGService
from app.schemas import (AskRequest, AskResponse, DocumentOut, EvidenceOut, HealthResponse,
                         IngestResponse, IngestResult, SearchHit, SearchRequest, SettingsIn,
                         SettingsOut)

router = APIRouter()


def _service(request: Request) -> RAGService:
    return request.app.state.rag


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(request: Request) -> HealthResponse:
    rag = _service(request)
    return HealthResponse(
        status="ok",
        documents=len(rag.store.documents()),
        chunks=len(rag.store),
        llm_ready=rag.llm is not None,
        llm_model=getattr(rag.llm, "model", None),
        embedding_model=rag.embedder.name,
    )


@router.post("/documents", response_model=IngestResponse, status_code=201, tags=["documents"])
async def ingest_documents(request: Request, files: list[UploadFile] = File(...)) -> IngestResponse:
    """Upload one or more PDF, TXT or Markdown files. Each file is reported separately."""
    rag = _service(request)
    limit = request.app.state.settings.max_upload_mb * 1024 * 1024
    results: list[IngestResult] = []
    for upload in files:
        name = (upload.filename or "untitled").replace("\\", "/").rsplit("/", 1)[-1]
        data = await upload.read(limit + 1)
        if len(data) > limit:
            results.append(IngestResult(name=name, status="failed",
                                        error=f"File is larger than {limit // (1024 * 1024)} MB."))
            continue
        try:
            info = await run_in_threadpool(rag.ingest, name, data)
        except IngestionError as exc:
            results.append(IngestResult(name=name, status="failed", error=str(exc)))
            continue
        results.append(IngestResult(
            name=name, status="replaced" if info["replaced"] else "ingested",
            id=info["id"], chunks=info["chunks"], pages=info["pages"],
        ))
    if all(r.status == "failed" for r in results):
        raise HTTPException(422, detail={"message": "No files could be ingested.",
                                         "results": [r.model_dump() for r in results]})
    return IngestResponse(results=results)


@router.get("/documents", response_model=list[DocumentOut], tags=["documents"])
def list_documents(request: Request) -> list[DocumentOut]:
    return [DocumentOut(**d) for d in _service(request).store.documents()]


@router.delete("/documents/{doc_id}", status_code=204, tags=["documents"])
def delete_document(doc_id: str, request: Request) -> None:
    if not _service(request).store.delete_document(doc_id):
        raise HTTPException(404, detail="Document not found.")


@router.post("/search", response_model=list[SearchHit], tags=["question answering"])
async def search(body: SearchRequest, request: Request) -> list[SearchHit]:
    """Raw semantic search, without the LLM. Useful for inspecting what the answerer will see."""
    hits = await run_in_threadpool(_service(request).retrieve, body.query, body.top_k)
    return [
        SearchHit(document=h.chunk.document, page=h.chunk.page, section=h.chunk.heading,
                  chunk_index=h.chunk.index, score=round(h.score, 4), text=h.chunk.text)
        for h in hits
    ]


@router.post("/ask", response_model=AskResponse, tags=["question answering"])
async def ask(body: AskRequest, request: Request) -> AskResponse:
    """Answer a question from the indexed documents, with cited evidence."""
    question = body.question.strip()
    if not question:
        raise HTTPException(422, detail="Question must not be blank.")
    try:
        result = await run_in_threadpool(_service(request).answer, question, body.top_k)
    except LLMNotConfiguredError as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    except LLMError as exc:
        raise HTTPException(502, detail=str(exc)) from exc
    return AskResponse(
        question=result.question,
        status=result.status,
        answer=result.answer,
        sources=[EvidenceOut(source=e.source, document=e.document, page=e.page, section=e.heading,
                             excerpt=e.excerpt, score=e.score, quote_verified=e.quote_verified)
                 for e in result.evidence],
        missing=result.missing,
        top_score=None if result.top_score is None else round(result.top_score, 4),
        latency_ms=result.latency_ms,
    )


def _settings_out(request: Request) -> SettingsOut:
    llm = _service(request).llm
    hint = getattr(request.app.state, "key_hint", None)
    return SettingsOut(provider="gemini" if isinstance(llm, GeminiAnswerGenerator) else
                       ("anthropic" if llm is not None else "none"),
                       model=getattr(llm, "model", None), configured=llm is not None, key_hint=hint)


@router.get("/settings", response_model=SettingsOut, tags=["system"])
def get_llm_settings(request: Request) -> SettingsOut:
    return _settings_out(request)


@router.post("/settings", response_model=SettingsOut, tags=["system"])
async def set_llm_settings(body: SettingsIn, request: Request) -> SettingsOut:
    """Set the Gemini API key and model at runtime. The key is verified with a tiny request and kept
    in server memory only (never written to disk or returned)."""
    key, model = body.api_key.strip(), body.model.strip()
    try:
        llm = GeminiAnswerGenerator(key, model)
        await run_in_threadpool(llm.check)
    except LLMError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    _service(request).llm = llm
    request.app.state.key_hint = key[-4:]
    return _settings_out(request)


@router.delete("/settings", response_model=SettingsOut, tags=["system"])
def clear_llm_settings(request: Request) -> SettingsOut:
    _service(request).llm = None
    request.app.state.key_hint = None
    return _settings_out(request)
