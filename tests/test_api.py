"""HTTP behaviour of the FastAPI app, including error responses."""

from app.llm import GeneratedAnswer, LLMError
from app.rag import UNAVAILABLE_MESSAGE
from tests.conftest import SAMPLE_DIR


def _upload(client, *files):
    return client.post("/api/documents", files=[("files", f) for f in files])


def _seed(client):
    files = [(p.name, p.read_bytes()) for p in sorted(SAMPLE_DIR.iterdir())]
    return _upload(client, *files)


def test_health_reports_state(client):
    body = client.get("/api/health").json()
    assert body == {"status": "ok", "documents": 0, "chunks": 0, "llm_ready": True,
                    "llm_model": "fake-llm", "embedding_model": "hashing-1024"}


def test_ui_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_upload_all_sample_formats(client):
    response = _seed(client)
    assert response.status_code == 201
    results = response.json()["results"]
    assert {r["status"] for r in results} == {"ingested"}
    assert {r["name"] for r in results} >= {"aero_x2_spec_sheet.pdf", "employee_handbook.md",
                                             "warranty_and_returns.txt"}
    pdf = next(r for r in results if r["name"].endswith(".pdf"))
    assert pdf["pages"] == 2

    listed = client.get("/api/documents").json()
    assert len(listed) == len(results)


def test_reupload_reports_replaced(client):
    _upload(client, ("a.txt", b"First version."))
    result = _upload(client, ("a.txt", b"Second version.")).json()["results"][0]
    assert result["status"] == "replaced"
    assert len(client.get("/api/documents").json()) == 1


def test_mixed_batch_reports_each_file(client):
    response = _upload(client, ("good.txt", b"Useful content here."), ("bad.png", b"\x89PNG"))
    assert response.status_code == 201
    statuses = {r["name"]: r["status"] for r in response.json()["results"]}
    assert statuses == {"good.txt": "ingested", "bad.png": "failed"}


def test_unsupported_and_empty_uploads_return_422(client):
    response = _upload(client, ("bad.png", b"\x89PNG"), ("empty.txt", b""))
    assert response.status_code == 422
    errors = [r["error"] for r in response.json()["detail"]["results"]]
    assert "Unsupported file type" in errors[0]
    assert "no extractable text" in errors[1]


def test_oversized_upload_is_rejected(client, settings):
    settings.max_upload_mb = 0
    response = _upload(client, ("big.txt", b"x" * 10))
    assert response.status_code == 422
    assert "larger than" in response.json()["detail"]["results"][0]["error"]


def test_upload_without_files_is_422(client):
    assert client.post("/api/documents").status_code == 422


def test_delete_document(client):
    doc_id = _upload(client, ("a.txt", b"Some text.")).json()["results"][0]["id"]
    assert client.delete(f"/api/documents/{doc_id}").status_code == 204
    assert client.get("/api/documents").json() == []
    assert client.delete(f"/api/documents/{doc_id}").status_code == 404


def test_search_returns_ranked_hits(client):
    _seed(client)
    hits = client.post("/api/search", json={"query": "parental leave weeks", "top_k": 3}).json()
    assert len(hits) == 3
    assert hits[0]["document"] == "employee_handbook.md"
    assert hits[0]["score"] >= hits[1]["score"] >= hits[2]["score"]


def test_ask_returns_answer_with_sources(client):
    _seed(client)
    response = client.post("/api/ask", json={"question": "How long is parental leave?"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["sources"][0]["document"]
    assert body["sources"][0]["excerpt"]
    assert body["latency_ms"] >= 0


def test_ask_unanswerable(client, fake_llm):
    _seed(client)
    fake_llm.response = GeneratedAnswer("unanswerable", "", [], "No CEO is mentioned.")
    body = client.post("/api/ask", json={"question": "Who is the CEO?"}).json()
    assert body["status"] == "unanswerable"
    assert body["answer"] == UNAVAILABLE_MESSAGE
    assert body["sources"] == []


def test_ask_before_any_upload(client):
    body = client.post("/api/ask", json={"question": "Anything?"}).json()
    assert body["status"] == "unanswerable"
    assert "No documents" in body["answer"]


def test_ask_validates_input(client):
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "   "}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/api/ask", json={"question": "ok", "top_k": 0}).status_code == 422
    assert client.post("/api/ask", json={}).status_code == 422


def test_ask_without_llm_is_503(client, rag):
    _seed(client)
    rag.llm = None
    response = client.post("/api/ask", json={"question": "battery"})
    assert response.status_code == 503
    assert "API key" in response.json()["detail"]


def test_ask_llm_failure_is_502(client, fake_llm):
    _seed(client)

    def boom(*_):
        raise LLMError("Could not reach the LLM provider.")

    fake_llm.generate = boom
    response = client.post("/api/ask", json={"question": "battery"})
    assert response.status_code == 502
    assert response.json()["detail"] == "Could not reach the LLM provider."


def test_settings_saves_verified_gemini_key(client, rag, monkeypatch):
    from app.llm import GeminiAnswerGenerator

    monkeypatch.setattr(GeminiAnswerGenerator, "check", lambda self: None)
    body = client.post("/api/settings", json={"api_key": "AIza-test-key-1234", "model": "gemini-3.5-flash-lite"}).json()
    assert body == {"provider": "gemini", "model": "gemini-3.5-flash-lite", "configured": True, "key_hint": "1234"}
    assert isinstance(rag.llm, GeminiAnswerGenerator)
    assert "AIza" not in client.get("/api/settings").text  # the key itself is never returned


def test_settings_rejects_bad_key_and_keeps_current_llm(client, rag, fake_llm, monkeypatch):
    from app.llm import GeminiAnswerGenerator

    def reject(self):
        raise LLMError("The Gemini API key was rejected. Check it in Settings.")

    monkeypatch.setattr(GeminiAnswerGenerator, "check", reject)
    response = client.post("/api/settings", json={"api_key": "AIza-wrong-key-0000"})
    assert response.status_code == 400
    assert "rejected" in response.json()["detail"]
    assert rag.llm is fake_llm


def test_settings_validation_and_removal(client, rag):
    assert client.post("/api/settings", json={"api_key": "short"}).status_code == 422
    body = client.delete("/api/settings").json()
    assert body["configured"] is False and rag.llm is None
