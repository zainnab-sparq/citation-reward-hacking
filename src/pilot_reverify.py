"""Re-pilot verification: consume LLM-extracted citations and run verification.

Phase A pilot's regex extractor missed mid-sentence and quoted-title citations.
Phase A.5 splits the pipeline:
  1. extract: LLM (Sonnet) reads generations.jsonl and writes extractions.jsonl
     with structured {raw_span, title, authors, year, venue, doi, case_name,
     reporter} records. (Done by AnthropicExtractorClient OR a Max subagent.)
  2. verify (this script): reads extractions.jsonl, verifies each citation
     against Crossref / S2 / OpenAlex / CourtListener, tabulates results.

Extractions JSONL format (one record per (question_id, condition)):
    {
      "question_id": "a-001",
      "condition": "citations_expected",
      "citations": [
        {"raw_span": "...", "title": "...", "authors": [...], "year": 2017,
         "venue": "...", "doi": null, "case_name": null, "reporter": null},
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.extract_citations import ExtractedCitation
from src.pilot import CONDITIONS, load_questions
from src.verify import VerificationCache, VerificationStatus, verify_citation


def _load_extractions(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


async def reverify(
    extractions_path: Path,
    cache_path: Path,
    questions_path: Path,
) -> dict[tuple[str, str], dict]:
    """Verify every extracted citation. Returns dict keyed by (question_id, condition)."""
    questions = load_questions(questions_path)
    qmap = {q.id: q for q in questions}
    cache = VerificationCache(cache_path)
    extractions = _load_extractions(extractions_path)

    results: dict[tuple[str, str], dict] = {}
    for record in extractions:
        qid = record["question_id"]
        condition = record["condition"]
        question = qmap[qid]
        statuses: list[str] = []
        details: list[dict] = []
        for cite_dict in record.get("citations", []):
            extracted = ExtractedCitation.from_dict(cite_dict)
            parsed = extracted.to_verify_citation(domain=question.domain)
            res = await verify_citation(
                extracted.raw_span,
                domain=question.domain,
                cache=cache,
                parsed=parsed,
            )
            statuses.append(res.status.value)
            details.append({
                "raw_span": extracted.raw_span[:80],
                "status": res.status.value,
                "sources": res.sources_checked,
            })
        results[(qid, condition)] = {
            "n_citations": len(statuses),
            "statuses": statuses,
            "fabricated": sum(1 for s in statuses if s == "FABRICATED"),
            "real": sum(1 for s in statuses if s == "REAL"),
            "unverifiable": sum(1 for s in statuses if s == "UNVERIFIABLE"),
            "details": details,
        }
    return results


def summarize(results: dict[tuple[str, str], dict], questions_path: Path) -> None:
    console = Console()
    questions = load_questions(questions_path)
    qmap = {q.id: q for q in questions}

    per_cond: dict[str, Counter] = defaultdict(Counter)
    per_cond_cite_count: dict[str, int] = defaultdict(int)
    for (_qid, cond), stats in results.items():
        per_cond[cond]["fabricated"] += stats["fabricated"]
        per_cond[cond]["real"] += stats["real"]
        per_cond[cond]["unverifiable"] += stats["unverifiable"]
        per_cond_cite_count[cond] += stats["n_citations"]

    table = Table(title="Re-pilot fabrication rates by prompt condition")
    table.add_column("Condition")
    table.add_column("# Citations")
    table.add_column("Real")
    table.add_column("Fabricated")
    table.add_column("Unverifiable")
    table.add_column("Fabrication rate")
    for cond, counts in per_cond.items():
        n = per_cond_cite_count[cond]
        rate = counts["fabricated"] / n if n else 0.0
        table.add_row(
            cond,
            str(n),
            str(counts["real"]),
            str(counts["fabricated"]),
            str(counts["unverifiable"]),
            f"{rate:.1%}",
        )
    console.print(table)

    detail = Table(title="Per-question detail (real/fab/unv)")
    detail.add_column("Q ID")
    detail.add_column("Domain")
    detail.add_column("Expected")
    for cond in [c.name for c in CONDITIONS]:
        detail.add_column(cond)
    for qid in sorted(qmap):
        q = qmap[qid]
        row = [qid, q.domain, q.expected_existence]
        for cond in [c.name for c in CONDITIONS]:
            stats = results.get((qid, cond), {})
            row.append(
                f"{stats.get('real', '-')}/{stats.get('fabricated', '-')}/{stats.get('unverifiable', '-')}"
            )
        detail.add_row(*row)
    console.print(detail)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/questions/pilot_15.jsonl"),
    )
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/verifications/pilot_cache_v2.jsonl"),
    )
    args = parser.parse_args()

    print(f"Verifying citations from {args.extractions}")
    results = await reverify(args.extractions, args.cache, args.questions)
    summarize(results, args.questions)


if __name__ == "__main__":
    asyncio.run(main())
