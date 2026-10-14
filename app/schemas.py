"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, examples=["How long is the battery warranty?"])
    top_k: int | None = Field(None, ge=1, le=20, description="Chunks to retrieve (defaults to TOP_K).")


class EvidenceOut(BaseModel):
    source: int = Field(description="Citation number used in the answer text, e.g. [1].")
    document: str
    page: int | None = None
    section: str | None = None
    excerpt: str
    score: float = Field(description="Cosine similarity between the question and this chunk.")
    quote_verified: bool = Field(description="True if the excerpt was found verbatim in the source.")


class AskResponse(BaseModel):
    question: str
    status: Literal["answered", "partial", "unanswerable"]
    answer: str
    sources: list[EvidenceOut]
    missing: str = Field("", description="What the documents did not cover, if anything.")
    top_score: float | None = None
    latency_ms: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=50)


class SearchHit(BaseModel):
    document: str
    page: int | None = None
    section: str | None = None
    chunk_index: int
    score: float
    text: str


class DocumentOut(BaseModel):
    id: str
    name: str
    chunks: int
    pages: int | None = None


class IngestResult(BaseModel):
    name: str
    status: Literal["ingested", "replaced", "failed"]
    id: str | None = None
    chunks: int = 0
    pages: int | None = None
    error: str | None = None


class IngestResponse(BaseModel):
    results: list[IngestResult]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    documents: int
    chunks: int
    llm_ready: bool
    llm_model: str | None
    embedding_model: str


class SettingsIn(BaseModel):
    api_key: str = Field(..., min_length=10, max_length=300)
    model: str = Field("gemini-3.5-flash-lite", min_length=1, max_length=100)


class SettingsOut(BaseModel):
    provider: str
    model: str | None
    configured: bool
    key_hint: str | None = Field(None, description="Last 4 characters of the key in use.")
