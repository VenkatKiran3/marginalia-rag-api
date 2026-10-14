"""Retrieval tests: the vector store itself, plus real semantic search with the production embedder."""

import numpy as np
import pytest

from app.embeddings import HashingEmbedder
from app.ingestion import Chunk
from app.rag import RAGService
from app.vector_store import VectorStore
from tests.conftest import SAMPLE_DIR


def _chunk(i: int, doc: str = "d") -> Chunk:
    return Chunk(id=f"{doc}-{i}", doc_id=doc, document=f"{doc}.txt", text=f"text {i}", index=i)


def test_store_returns_hits_in_descending_score_order():
    store = VectorStore()
    vectors = np.eye(4, dtype=np.float32)
    store.add([_chunk(i) for i in range(4)], vectors, "test")
    query = np.array([0.1, 0.9, 0.5, 0.0], dtype=np.float32)
    hits = store.search(query, k=3)
    assert [c.index for c, _ in hits] == [1, 2, 0]
    assert hits[0][1] >= hits[1][1] >= hits[2][1]


def test_store_handles_k_larger_than_index_and_empty_index():
    store = VectorStore()
    assert store.search(np.ones(3), k=5) == []
    store.add([_chunk(0)], np.ones((1, 3)), "test")
    assert len(store.search(np.ones(3), k=5)) == 1


def test_store_delete_and_persistence_roundtrip(tmp_path):
    store = VectorStore(tmp_path)
    store.add([_chunk(0, "a"), _chunk(1, "a")], np.eye(2), "test")
    store.add([_chunk(0, "b")], np.array([[1.0, 0.0]]), "test")
    assert store.delete_document("a") == 2

    reloaded = VectorStore(tmp_path)
    assert len(reloaded) == 1
    assert reloaded.documents()[0]["id"] == "b"
    assert reloaded.search(np.array([1.0, 0.0]), k=1)[0][0].doc_id == "b"


def test_store_refuses_to_mix_embedding_models():
    store = VectorStore()
    store.add([_chunk(0)], np.ones((1, 2)), "model-a")
    with pytest.raises(ValueError, match="re-index"):
        store.add([_chunk(1)], np.ones((1, 2)), "model-b")


def test_reingesting_a_file_replaces_its_chunks(rag):
    rag.ingest("notes.txt", b"The launch date is in June. " * 5)
    info = rag.ingest("notes.txt", b"The launch date moved to September.")
    assert info["replaced"] is True
    assert len(rag.store.documents()) == 1
    assert all("September" in h.chunk.text for h in rag.retrieve("launch date"))


# ------------------------------------------------------------------ semantic search (real embedder)

@pytest.fixture(scope="module")
def semantic_rag():
    try:
        from app.embeddings import FastEmbedEmbedder

        embedder = FastEmbedEmbedder("BAAI/bge-small-en-v1.5", cache_dir="data/models")
    except Exception as exc:  # model download needs network on first run
        pytest.skip(f"FastEmbed model unavailable: {exc}")
    service = RAGService(VectorStore(), embedder, llm=None)
    for path in sorted(SAMPLE_DIR.iterdir()):
        service.ingest(path.name, path.read_bytes())
    return service


@pytest.mark.parametrize("query, expected_doc", [
    # Paraphrased: little or no word overlap with the source text.
    ("How much does the bicycle weigh?", "aero_x2_spec_sheet.pdf"),
    ("Can I take a holiday carry-over into next year?", "employee_handbook.md"),
    ("What happens if my battery degrades quickly?", "warranty_and_returns.txt"),
    ("Which update lets me stop charging early to protect the cells?", "firmware_release_notes.md"),
    ("How many weeks off do new parents get?", "employee_handbook.md"),
])
def test_semantic_search_finds_paraphrased_questions(semantic_rag, query, expected_doc):
    top3 = [h.chunk.document for h in semantic_rag.retrieve(query, top_k=3)]
    assert expected_doc in top3


def test_semantic_scores_separate_off_topic_questions(semantic_rag):
    on_topic = semantic_rag.retrieve("How long is the battery warranty?")[0].score
    off_topic = semantic_rag.retrieve("What is the capital of Australia?")[0].score
    assert on_topic > 0.6
    assert off_topic < 0.5 <= on_topic  # 0.5 is the default MIN_RELEVANCE gate


def test_hashing_embedder_is_deterministic_and_normalised():
    embedder = HashingEmbedder()
    a, b = embedder.embed_query("battery warranty"), embedder.embed_query("battery warranty")
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0)
