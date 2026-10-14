"""Evaluate retrieval and end-to-end answering on eval/questions.json.

    python -m eval.run_eval                # retrieval metrics always; answer metrics if ANTHROPIC_API_KEY is set
    python -m eval.run_eval --retrieval-only

Indexes sample_docs/ into a throwaway in-memory store, so it never touches your real index.
Writes eval/results.md (human-readable) and eval/results.json (raw).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.config import get_settings
from app.main import build_service
from app.vector_store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = json.loads((ROOT / "eval" / "questions.json").read_text(encoding="utf-8"))


def _norm(text: str) -> str:
    return text.lower().replace(",", "").replace(" ", "")


def run(retrieval_only: bool) -> dict:
    settings = get_settings()
    rag = build_service(settings, store=VectorStore())
    if retrieval_only:
        rag.llm = None
    for path in sorted((ROOT / "sample_docs").iterdir()):
        rag.ingest(path.name, path.read_bytes())

    rows = []
    for q in QUESTIONS:
        hits = rag.retrieve(q["question"], settings.top_k)
        retrieved = {h.chunk.document for h in hits}
        expected = set(q["expected_sources"])
        row = {
            "id": q["id"], "category": q["category"], "question": q["question"],
            "top_score": round(hits[0].score, 3) if hits else None,
            "retrieval_recall": len(expected & retrieved) / len(expected) if expected else None,
            "gated": not hits or hits[0].score < settings.min_relevance,
        }
        if rag.llm is not None:
            result = rag.answer(q["question"])
            cited = {e.document for e in result.evidence}
            row.update({
                "status": result.status,
                "status_ok": result.status in q["expected_status"],
                "fact_recall": (sum(_norm(f) in _norm(result.answer) for f in q["expected_facts"])
                                / len(q["expected_facts"])) if q["expected_facts"] else None,
                "citation_recall": len(expected & cited) / len(expected) if expected else None,
                "quotes_verified": f"{sum(e.quote_verified for e in result.evidence)}/{len(result.evidence)}",
                "answer": result.answer,
                "missing": result.missing,
                "sources": sorted(cited),
                "latency_ms": result.latency_ms,
            })
        rows.append(row)
        print(f"{q['id']:>3}  {row.get('status', '-'):<13} top={row['top_score']}  {q['question']}")

    return {"llm_model": getattr(rag.llm, "model", None), "embedding_model": rag.embedder.name,
            "top_k": settings.top_k, "min_relevance": settings.min_relevance, "rows": rows}


def _mean(values) -> str:
    values = [v for v in values if v is not None]
    return f"{sum(values) / len(values):.0%}" if values else "n/a"


def write_report(report: dict) -> str:
    rows = report["rows"]
    end_to_end = report["llm_model"] is not None
    lines = [
        "# Evaluation results", "",
        f"- Embedding model: `{report['embedding_model']}`, top-k {report['top_k']}, "
        f"relevance gate {report['min_relevance']}",
        f"- LLM: `{report['llm_model']}`" if end_to_end else "- LLM: not run (retrieval-only)",
        "", "## Summary", "",
        "| Category | Questions | Retrieval recall@k |" + (" Status correct | Fact recall | Citation recall |" if end_to_end else ""),
        "|---|---|---|" + ("---|---|---|" if end_to_end else ""),
    ]
    for category in ["answerable", "multi_evidence", "unanswerable"]:
        group = [r for r in rows if r["category"] == category]
        line = f"| {category} | {len(group)} | {_mean(r['retrieval_recall'] for r in group)} |"
        if end_to_end:
            line += (f" {_mean(float(r['status_ok']) for r in group)} | {_mean(r['fact_recall'] for r in group)}"
                     f" | {_mean(r['citation_recall'] for r in group)} |")
        lines.append(line)

    lines += ["", "## Per question", "",
              "| ID | Question | Top score | Retrieval recall | Relevance gate |" + (" Status | Facts | Cited docs | Quotes verified |" if end_to_end else ""),
              "|---|---|---|---|---|" + ("---|---|---|---|" if end_to_end else "")]
    for r in rows:
        line = f"| {r['id']} | {r['question']} | {r['top_score']} | {_mean([r['retrieval_recall']])} | {'blocked' if r['gated'] else 'passed'} |"
        if end_to_end:
            line += (f" {r['status']} {'✓' if r['status_ok'] else '✗'} | {_mean([r['fact_recall']])} |"
                     f" {', '.join(r['sources']) or '—'} | {r['quotes_verified']} |")
        lines.append(line)

    if end_to_end:
        lines += ["", "## Answers", ""]
        for r in rows:
            lines += [f"**{r['id']}. {r['question']}**", "", f"> {r['answer']}"]
            if r["missing"]:
                lines += ["", f"_Missing:_ {r['missing']}"]
            lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retrieval-only", action="store_true", help="Skip the LLM even if a key is set.")
    args = parser.parse_args()
    report = run(args.retrieval_only)
    (ROOT / "eval" / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "eval" / "results.md").write_text(write_report(report), encoding="utf-8")
    print("\nWrote eval/results.md and eval/results.json")


if __name__ == "__main__":
    main()
