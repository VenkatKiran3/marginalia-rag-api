import pytest

from app.ingestion import IngestionError, chunk_document, parse_document, split_text
from tests.conftest import SAMPLE_DIR


def test_split_text_respects_size_and_overlaps():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(80))
    chunks = split_text(text, chunk_size=300, overlap=80)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    # Consecutive chunks share trailing/leading sentences.
    for previous, current in zip(chunks, chunks[1:]):
        assert current.split(".")[0] in previous


def test_split_text_breaks_oversized_sentences():
    chunks = split_text("word " * 500, chunk_size=200, overlap=20)
    assert all(len(c) <= 200 for c in chunks)


def test_split_text_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("hello", chunk_size=100, overlap=100)


def test_markdown_sections_carry_heading_breadcrumbs():
    md = b"# Guide\n\nIntro.\n\n## Setup\n\nInstall it.\n\n### Docker\n\nRun compose.\n\n## Usage\n\nAsk."
    headings = [s.heading for s in parse_document("guide.md", md)]
    assert headings == ["Guide", "Guide › Setup", "Guide › Setup › Docker", "Guide › Usage"]


def test_markdown_ignores_hash_lines_inside_code_blocks():
    md = b"# Title\n\n```bash\n# not a heading\necho hi\n```\n"
    sections = parse_document("code.md", md)
    assert len(sections) == 1 and "# not a heading" in sections[0].text


def test_pdf_chunks_record_page_numbers():
    chunks = chunk_document("spec.pdf", (SAMPLE_DIR / "aero_x2_spec_sheet.pdf").read_bytes(), 700, 120)
    assert {c.page for c in chunks} == {1, 2}
    assert any("21.4 kg" in c.text for c in chunks if c.page == 2)


def test_txt_with_windows_encoding_is_decoded():
    sections = parse_document("legacy.txt", "Café prices: €5".encode("cp1252"))
    assert "Café" in sections[0].text


def test_chunk_ids_are_stable_per_filename():
    first = chunk_document("notes.txt", b"Alpha. Beta.", 700, 120)
    second = chunk_document("notes.txt", b"Gamma. Delta.", 700, 120)
    assert first[0].doc_id == second[0].doc_id


@pytest.mark.parametrize("filename", ["image.png", "archive.zip", "noextension"])
def test_unsupported_file_types_are_rejected(filename):
    with pytest.raises(IngestionError, match="Unsupported file type"):
        parse_document(filename, b"data")


@pytest.mark.parametrize("payload", [b"", b"   \n\n  "])
def test_empty_documents_are_rejected(payload):
    with pytest.raises(IngestionError, match="no extractable text"):
        chunk_document("empty.txt", payload, 700, 120)


def test_corrupt_pdf_is_rejected():
    with pytest.raises(IngestionError):
        parse_document("broken.pdf", b"%PDF-1.7 this is not really a pdf")
