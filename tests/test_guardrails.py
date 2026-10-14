"""Failure cases and reliability guardrails of the answering pipeline."""

import pytest

from app.llm import GeneratedAnswer, LLMError, Quote, parse_answer
from app.rag import (NO_DOCUMENTS_MESSAGE, UNAVAILABLE_MESSAGE, LLMNotConfiguredError,
                     strip_invalid_citations, verify_quote)


def test_answer_includes_cited_evidence(seeded_rag):
    result = seeded_rag.answer("What is the battery warranty?")
    assert result.status == "answered"
    assert result.evidence and result.evidence[0].source == 1
    assert result.evidence[0].quote_verified is True
    assert result.evidence[0].document.endswith((".txt", ".md", ".pdf"))


def test_empty_index_reports_no_documents_without_calling_llm(rag, fake_llm):
    result = rag.answer("Anything?")
    assert result.status == "unanswerable"
    assert result.answer == NO_DOCUMENTS_MESSAGE
    assert fake_llm.calls == []


def test_low_relevance_short_circuits_before_llm(seeded_rag, fake_llm):
    seeded_rag.min_relevance = 0.99
    result = seeded_rag.answer("What is the capital of Australia?")
    assert result.status == "unanswerable"
    assert result.answer == UNAVAILABLE_MESSAGE
    assert result.evidence == []
    assert fake_llm.calls == []


def test_model_saying_unanswerable_yields_standard_message(seeded_rag, fake_llm):
    fake_llm.response = GeneratedAnswer("unanswerable", "", [], "The CEO is not named.")
    result = seeded_rag.answer("Who is the CEO?")
    assert result.status == "unanswerable"
    assert result.answer == UNAVAILABLE_MESSAGE
    assert result.missing == "The CEO is not named."
    assert result.evidence == []


def test_answer_without_any_valid_citation_is_withheld(seeded_rag, fake_llm):
    fake_llm.response = GeneratedAnswer("answered", "The CEO is Jane Doe [42].", [Quote(42, "Jane Doe")])
    result = seeded_rag.answer("Who is the CEO?")
    assert result.status == "unanswerable"
    assert "Jane Doe" not in result.answer


def test_hallucinated_citation_numbers_are_removed(seeded_rag, fake_llm):
    fake_llm.response = GeneratedAnswer("answered", "Fact one [1]. Fact two [9].", [Quote(1, "x")])
    result = seeded_rag.answer("battery warranty")
    assert result.answer == "Fact one [1]. Fact two."
    assert [e.source for e in result.evidence] == [1]


def test_unverifiable_quote_falls_back_to_source_text(seeded_rag, fake_llm):
    fake_llm.response = GeneratedAnswer("answered", "Claim [1].", [Quote(1, "A sentence the source never says.")])
    result = seeded_rag.answer("battery warranty")
    evidence = result.evidence[0]
    assert evidence.quote_verified is False
    assert "never says" not in evidence.excerpt
    assert evidence.excerpt  # the retrieved chunk text is shown instead


def test_partial_answers_keep_missing_explanation(seeded_rag, fake_llm):
    fake_llm.response = GeneratedAnswer("partial", "Price is EUR 3,290 [1].", [Quote(1, "x")],
                                        "No US price is listed.")
    result = seeded_rag.answer("What does it cost in dollars?")
    assert result.status == "partial"
    assert result.missing == "No US price is listed."


def test_missing_llm_raises_clear_error(seeded_rag):
    seeded_rag.llm = None
    with pytest.raises(LLMNotConfiguredError):
        seeded_rag.answer("battery warranty")


def test_llm_errors_propagate(seeded_rag, fake_llm):
    def boom(*_):
        raise LLMError("provider down")

    fake_llm.generate = boom
    with pytest.raises(LLMError):
        seeded_rag.answer("battery warranty")


@pytest.mark.parametrize("quote, expected", [
    ("holds less than 70% of its rated capacity", True),
    ("HOLDS   less than 70%\nof its rated capacity", True),   # whitespace / case noise
    ("holds less than 70 % of its rated capacity", True),     # PDF-style spacing noise
    ("holds less than 50% of its rated capacity", True),      # one-char drift is tolerated...
    ("The battery is guaranteed for ten years", False),        # ...a fabricated claim is not
    ("", False),
])
def test_verify_quote(quote, expected):
    source = "A battery is considered defective if it holds less than 70% of its rated capacity."
    assert verify_quote(quote, source) is expected


def test_strip_invalid_citations_handles_grouped_markers():
    text, used = strip_invalid_citations("A [1, 7]. B [3]. C [2,3].", {1, 2, 3})
    assert text == "A [1]. B [3]. C [2, 3]."
    assert used == {1, 2, 3}


def test_parse_answer_rejects_malformed_json():
    with pytest.raises(LLMError):
        parse_answer("not json")
    with pytest.raises(LLMError):
        parse_answer('{"answer": "missing status"}')
