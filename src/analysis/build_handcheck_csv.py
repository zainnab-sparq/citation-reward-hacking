"""Build the §8 A.5 hand-check CSV from Step 0 Phase B pilot artifacts.

Joins generations + extractions + verifications + questions into one row
per citation. Each row carries the model's output excerpt around the cited
span so the hand-checker can judge whether the verifier's label is right
without flipping between files.

Columns (in order):
  row_id, question_id, domain, cell, condition, citation_index,
  raw_span, title, case_name, year, venue, doi,
  verifier_status, output_excerpt,
  agree (y/n — empty for user), correct_status (only if disagree), notes

The hand-check gate per pre-reg §8 A.5 requires ≥90% agreement on a
≥30-citation sample. Step 0's full 50-citation set is small enough to
hand-check in entirety; if a stratified subsample is preferred, the
caller can filter the output CSV downstream.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def _load(p: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _excerpt(text: str, span: str, *, window: int = 250) -> str:
    """Return ±window characters around the citation span in the output text.

    Falls back to the head of the text when the span isn't found verbatim
    (the LLM extractor sometimes returns a normalized form rather than the
    exact substring).
    """
    if not text:
        return ""
    if not span:
        return text[: 2 * window]
    pos = text.find(span)
    if pos < 0:
        # Try a looser match: just the case name / first 30 chars
        snippet = span[:30]
        pos = text.find(snippet)
        if pos < 0:
            return text[: 2 * window]
    start = max(0, pos - window)
    end = min(len(text), pos + len(span) + window)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return prefix + text[start:end] + suffix


def build_rows(
    *,
    generations: Iterable[dict[str, Any]],
    extractions: Iterable[dict[str, Any]],
    verifications: Iterable[dict[str, Any]],
    questions: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One row per (question_id, condition, model_id, trial_id, citation_index)."""
    gen_by_key = {
        (g["question_id"], g["condition"], g["model_id"], int(g.get("trial_id", 0))): g
        for g in generations
    }
    ext_by_key = {
        (e["question_id"], e["condition"], e["model_id"], int(e.get("trial_id", 0))): e
        for e in extractions
    }
    rows: list[dict[str, Any]] = []
    row_id = 0
    for v in verifications:
        key = (v["question_id"], v["condition"], v["model_id"], int(v.get("trial_id", 0)))
        ext = ext_by_key.get(key)
        gen = gen_by_key.get(key)
        if not ext or not gen:
            continue
        q = questions.get(v["question_id"], {})
        citations = ext.get("citations", [])
        for idx, status in enumerate(v.get("citation_statuses", [])):
            row_id += 1
            cite = citations[idx] if idx < len(citations) else {}
            rows.append(
                {
                    "row_id": row_id,
                    "question_id": v["question_id"],
                    "domain": q.get("domain", ""),
                    "cell": q.get("cell", ""),
                    "condition": v["condition"],
                    "citation_index": idx,
                    "raw_span": cite.get("raw_span", ""),
                    "title": cite.get("title", ""),
                    "case_name": cite.get("case_name", "") or "",
                    "year": cite.get("year", "") or "",
                    "venue": cite.get("venue", "") or "",
                    "doi": cite.get("doi", "") or "",
                    "verifier_status": status,
                    "output_excerpt": _excerpt(gen.get("output_text", ""), cite.get("raw_span", "")),
                    "agree": "",  # y / n — fill at hand-check
                    "correct_status": "",  # only if disagree: REAL_CONFIRMED / FABRICATED / MISGROUNDED / UNVERIFIABLE
                    "notes": "",
                }
            )
    return rows


_COLUMNS = [
    "row_id",
    "question_id",
    "domain",
    "cell",
    "condition",
    "citation_index",
    "raw_span",
    "title",
    "case_name",
    "year",
    "venue",
    "doi",
    "verifier_status",
    "output_excerpt",
    "agree",
    "correct_status",
    "notes",
]


def write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in _COLUMNS})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.analysis.build_handcheck_csv",
        description="Build the §8 A.5 hand-check CSV from Step 0 artifacts.",
    )
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--verifications", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    questions = {q["id"]: q for q in _load(args.questions)}
    rows = build_rows(
        generations=_load(args.generations),
        extractions=_load(args.extractions),
        verifications=_load(args.verifications),
        questions=questions,
    )
    write_csv(rows, args.output)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
