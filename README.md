# Marginalia — a reliable RAG API

Ask questions of your own documents (PDF, TXT, Markdown). Every answer cites the passages it came from, and when the documents don't contain the answer, the service says so instead of guessing.

- **FastAPI** service with ingestion, search and question-answering endpoints
- **Local embeddings** (`BAAI/bge-small-en-v1.5` via FastEmbed/ONNX, no API key needed) and a persistent NumPy vector store
- **Google Gemini** (`gemini-3.5-flash-lite` by default) for grounded answers, with JSON-schema-constrained output. Paste the API key in the UI (gear icon), or set `GEMINI_API_KEY`. Claude is supported as an alternative provider.
- **Four reliability gates** between retrieval and the response (see below)
- A web UI at `/` where answers show their sources as margin notes
- 60 automated tests and a 13-question evaluation set

## Quick start

### Docker

```bash
docker compose up --build     # optional: put GEMINI_API_KEY in .env, or paste the key in the UI
```

Open http://localhost:8000 for the UI or http://localhost:8000/docs for the API reference. The sample documents in `sample_docs/` are indexed on first start. The embedding model is built into the image, so the container runs without network access except for LLM calls.

### Local (Python 3.11+)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Open **http://localhost:8000**, click the **gear icon** (bottom left), paste your Gemini API key from https://aistudio.google.com/apikey and click **Save and test**. The key is checked with a small request, kept in server memory only, and optionally remembered in your browser, so it reconnects after a server restart.

```bash
```

On first run the embedding model (~70 MB) is downloaded to `data/models/`. Without a key, ingestion and search still work, and `/api/ask` returns `503` with a clear message.

### Tests and evaluation

```bash
pytest                         # 63 tests, ~3 s, no API key needed
GEMINI_API_KEY=... python -m eval.run_eval   # retrieval metrics; adds answer metrics when a key is set
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/documents` | Upload one or more files (`multipart/form-data`, field `files`). Each file gets its own result; re-uploading a filename replaces it. |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `POST` | `/api/ask` | `{"question": "...", "top_k": 8}` → answer, status, sources |
| `POST` | `/api/search` | Raw semantic search, no LLM, for inspecting retrieval |
| `GET` | `/api/health` | Index size and whether the LLM is configured |
| `GET`/`POST`/`DELETE` | `/api/settings` | Read, set (`{"api_key", "model"}`, verified before saving) or clear the Gemini key at runtime |

```bash
curl -F "files=@sample_docs/warranty_and_returns.txt" localhost:8000/api/documents
curl -X POST localhost:8000/api/ask -H "Content-Type: application/json" \
     -d '{"question": "How long is the battery warranty if I register my bike?"}'
```

```json
{
  "question": "How long is the battery warranty if I register my bike?",
  "status": "answered",
  "answer": "If you register within 60 days of purchase, the battery warranty extends to 3 years [1]...",
  "sources": [{
    "source": 1,
    "document": "warranty_and_returns.txt",
    "page": null,
    "section": null,
    "excerpt": "receive an extended battery warranty of 3 years",
    "score": 0.79,
    "quote_verified": true
  }],
  "missing": "",
  "top_score": 0.79,
  "latency_ms": 2140
}
```

`status` is `answered`, `partial` (some of the question is covered; `missing` says what isn't) or `unanswerable` (`answer` is a fixed sentence: *"The documents don't contain information to answer this question."*, and `sources` is empty).

**Error codes:** `422` invalid input, unsupported, empty, corrupt or oversized files; `400` Gemini key or model rejected in Settings; `404` unknown document; `503` no LLM configured; `502` LLM provider failure.

## Architecture

```
app/
  config.py        settings from environment / .env
  ingestion.py     PDF/TXT/MD parsing → sections → overlapping chunks
  embeddings.py    FastEmbed (production) and hashing (tests) embedders
  vector_store.py  persistent exact-cosine vector store
  llm.py           prompt, JSON schema, Gemini and Claude adapters
  rag.py           pipeline + reliability gates
  api.py           FastAPI routes;  schemas.py  request/response models
  main.py          app factory and wiring;  static/index.html  UI
eval/              questions.json, run_eval.py, results.md, test_results.txt
sample_docs/       4 fictional documents (PDF, TXT, 2× MD)
tests/             ingestion, retrieval, guardrails, API
```

**Ingestion.** PDFs are split per page, so every chunk keeps its page number. Markdown is split at headings, so every chunk keeps its section path (e.g. `Handbook › Time off › Parental leave`), and the heading is prepended when embedding to give short chunks context. Text is packed sentence by sentence into ~700-character chunks with ~120 characters of overlap, so a fact is never cut in half at a boundary.

**Retrieval.** `bge-small-en-v1.5` is small, CPU-fast, and strong on retrieval benchmarks; it runs locally, so documents are never sent to a third party for embedding. Queries use the model's query instruction (asymmetric search). The vector store does exact cosine search over a NumPy matrix, persisted to `data/index/`. At this scale (up to tens of thousands of chunks) exact search takes milliseconds, is deterministic, and needs no extra service. The store exposes add/delete/search, so it can be swapped for Qdrant or pgvector without touching the pipeline.

**Reliability: four gates.** Each gate can only make an answer more conservative.

1. **Relevance gate.** If the best chunk scores below `MIN_RELEVANCE` (0.5), the LLM is never called. This catches off-topic questions cheaply. The threshold was calibrated on the eval set: off-topic questions score about 0.45, answerable ones 0.58–0.86.
2. **Grounded prompt plus structured output.** The model sees only numbered excerpts, must cite each sentence, must return `answered`, `partial` or `unanswerable`, and is told to treat excerpt text as data, not instructions. Output is constrained by a JSON schema, so it always parses.
3. **Citation check.** Citations to sources the model was never shown are removed. An answer left with no valid citation is withheld and returned as unanswerable.
4. **Quote verification.** Each evidence quote must appear in its source chunk, allowing for whitespace, case and PDF-extraction noise. If a quote can't be matched, the UI and API show the retrieved passage itself (`quote_verified: false`), never the model's wording.

In-domain unanswerable questions (e.g. "Who is the CEO?") score as high as answerable ones (0.64–0.70), so a similarity threshold alone can't catch them. Gates 2–4 handle those.

**LLM.** Gemini through the official `google-genai` SDK with `response_json_schema` so the output always matches the answer schema. Errors are translated into plain messages (rejected key, unknown model, quota exceeded, network). Claude (`LLM_PROVIDER=anthropic`) is available as an alternative. The `AnswerGenerator` protocol keeps the provider swappable; tests use a scripted fake.

## Configuration

All settings are environment variables; see `.env.example`. The main ones are `GEMINI_API_KEY`, `GEMINI_MODEL`, `LLM_PROVIDER`, `TOP_K` (8), `MIN_RELEVANCE` (0.5), `CHUNK_SIZE` / `CHUNK_OVERLAP`, `DATA_DIR`, `SEED_DIR` and `MAX_UPLOAD_MB`.

## Evaluation

`eval/questions.json` has 13 questions about the sample documents (a fictional e-bike company, so answers can't come from the model's general knowledge):

| Category | Count | Example |
|---|---|---|
| Answerable | 5 | "Can I send my bike back if I've already ridden it 80 km?" |
| Multiple pieces of evidence | 4 | "How long does it take to charge to 80%, and which firmware version added the 80% charge limit?" (PDF + release notes) |
| Unanswerable | 4 | "Who is the CEO?", "What does it cost in US dollars?" (only a EUR price exists), an off-topic question |

The runner scores retrieval recall@k, status accuracy, fact recall (expected facts present in the answer), citation recall and quote verification. Results are written to `eval/results.md`.

**Results (`gemini-3.5-flash-lite`, full report in [`eval/results.md`](eval/results.md)).** 12 of 13 questions got the correct status.

| Category | Status correct | Fact recall | Citation recall | Retrieval recall@8 |
|---|---|---|---|---|
| Answerable (5) | 100% | 100% | 100% | 100% |
| Multiple evidence (4) | 75% | 88% | 88% | 100% |
| Unanswerable (4) | 100% | n/a | n/a | n/a |

- Every unanswerable question was declined, including the in-domain ones ("Who is the CEO?", "Is there a GPS tracker?") and the trap question asking for a US dollar price when only a EUR price exists. The off-topic question was blocked by the relevance gate before reaching the LLM.
- All 13 evidence quotes were verified verbatim against their source chunks.
- Multi-document answers combined sources correctly. For example, M3 combined the handbook's "15% of retail price" with the spec sheet's EUR 3,290 and calculated EUR 493.50.
- **The one miss (M2)** asked about warranty length and "the battery's rated capacity" without naming the bike. The model answered the warranty part (3 years), marked the answer `partial`, and declined to apply the Aero X2's 540 Wh to "my bike". The question is ambiguous, and the model chose the conservative reading. We kept the question unchanged rather than rewriting it to pass.

**Tests.** 63 pass (`eval/test_results.txt`). They cover ingestion (all three formats, chunk sizes, overlap, encodings, empty, corrupt and unsupported files), the vector store (ranking, deletion, persistence, mixing embedding models), real semantic retrieval on paraphrased questions, every guardrail, and API behaviour including the 422/404/502/503 paths.

## Limitations

- **The LLM is the last line of defence for in-domain unanswerable questions.** Gates 3–4 catch fabricated citations and quotes, but not a correctly cited sentence that is overstated in the answer text. An NLI or LLM-judge entailment check per sentence would close this.
- **Retrieval is dense-only.** Exact identifiers such as part numbers or error codes would benefit from hybrid BM25 + vector search and a cross-encoder reranker. During development, a two-part question ranked a needed chunk 6th, which is why `TOP_K` is 8 rather than 5.
- **Doc-level eval metrics can hide chunk-level misses.** Retrieval recall is measured per document, which is how the case above slipped past it at first. A chunk-level gold set would be stricter.
- **The eval set is small** (13 questions, 4 documents). It shows the behaviours but isn't statistically meaningful.
- **No OCR.** Scanned PDFs without a text layer are rejected as having no extractable text. Tables and multi-column layouts lose some structure.
- **Single-process store.** One writer, in-memory search. Run several replicas with a shared vector database instead.
- **No authentication, rate limiting or multi-tenancy.** Anyone who can reach the server can change its API key via `/api/settings`. Run it locally or behind a gateway.
- **The browser-remembered key is stored in `localStorage`.** Untick "Remember the key" on shared computers.
- **Windows note.** `onnxruntime` needs a recent MSVC runtime. `requirements.txt` installs `msvc-runtime` on Windows, and the app preloads it. Docker/Linux is unaffected.
