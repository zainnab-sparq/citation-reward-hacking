"""CLI wrapper around CandidateRecruiter. Pulls N candidates per (cell, domain)
from OpenAlex, writes raw-candidate JSONL.

For legal, see recruit_legal_cli.py (CourtListener-based).
For unanswerable cells, see construct_unanswerable.py (LLM-driven).

Output JSONL schema (one record per candidate):
    {
      "id": "<domain>_<cell>_<index>",
      "domain": "academic|medical",
      "cell": "<cell>",
      "expected_existence": "answerable",
      "expected_citation": {
        "title": "...",
        "authors": ["..."],
        "year": 2020,
        "venue": "...",
        "doi": "10.../...",
        "openalex_id": "https://openalex.org/W...",
        "cited_by_count": 12
      },
      "metadata": {
        "recruited_via": "openalex",
        "recruited_at": "2026-05-03T...",
        "filter": "...",
        "seed": 42
      }
    }

Run:
    OPENALEX_MAILTO=you@example.com .venv/Scripts/python.exe \\
        -m src.dataset_construction.recruit_cli \\
        --cell obscure_real --domain academic --n 250 \\
        --out data/questions/raw/academic_obscure_real.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from src.dataset_construction.openalex_client import OpenAlexClient, Work
from src.dataset_construction.recruiter import CandidateRecruiter, _CELL_FILTERS


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _clean_title(t: str) -> str:
    """OpenAlex titles occasionally embed JATS XML or HTML tags."""
    return _HTML_TAG_RE.sub("", t).strip()


def _record(work: Work, domain: str, cell: str, index: int, filter_str: str, seed: int) -> dict:
    return {
        "id": f"{domain}_{cell}_{index:04d}",
        "domain": domain,
        "cell": cell,
        "expected_existence": "answerable",
        "expected_citation": {
            "title": _clean_title(work.title),
            "authors": [work.first_author] if work.first_author else [],
            "year": work.publication_year,
            "venue": work.venue,
            "doi": work.doi,
            "openalex_id": work.id,
            "cited_by_count": work.cited_by_count,
        },
        "metadata": {
            "recruited_via": "openalex",
            "recruited_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "filter": filter_str,
            "seed": seed,
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    mailto = os.environ.get("OPENALEX_MAILTO")
    if not mailto:
        raise SystemExit("set OPENALEX_MAILTO env var")

    oa = OpenAlexClient(mailto=mailto)
    rec = CandidateRecruiter(oa)
    works = await rec.recruit(cell=args.cell, domain=args.domain, n=args.n, seed=args.seed)
    filter_str = ",".join(_CELL_FILTERS[(args.cell, args.domain)])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, work in enumerate(works):
            record = _record(work, args.domain, args.cell, i, filter_str, args.seed)
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written} candidates to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
