"""CLI: recruit legal candidates via CourtListener; write JSONL.

Run:
    .venv/Scripts/python.exe -m src.dataset_construction.recruit_legal_cli \\
        --cell known_real_scotus --n 250 --out data/questions/raw/legal_known_real_scotus.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from src.dataset_construction.courtlistener_recruiter import (
    LegalCandidate,
    recruit_circuit,
    recruit_district_obscure,
    recruit_scotus,
)


_CELL_DESCRIPTIONS = {
    "known_real_scotus": "SCOTUS opinions with citeCount >= 100",
    "known_real_circuit": "Federal Circuit Courts of Appeals opinions, citeCount >= 50",
    "obscure_real_district": "Federal District Court opinions, citeCount in [2, 49]",
}


async def _do_recruit(cell: str, n: int) -> list[LegalCandidate]:
    if cell == "known_real_scotus":
        return await recruit_scotus(n)
    if cell == "known_real_circuit":
        return await recruit_circuit(n)
    if cell == "obscure_real_district":
        return await recruit_district_obscure(n)
    raise ValueError(f"unknown legal cell: {cell}")


def _record(cand: LegalCandidate, cell: str, index: int) -> dict:
    return {
        "id": f"legal_{cell}_{index:04d}",
        "domain": "legal",
        "cell": cell,
        "expected_existence": "answerable",
        "expected_citation": {
            "title": cand.case_name,
            "case_name": cand.case_name,
            "reporter": cand.citation[0] if cand.citation else None,
            "year": int(cand.date_filed.split("-")[0]) if cand.date_filed else None,
            "venue": cand.court,
            "court_id": cand.court_id,
            "courtlistener_cluster_id": cand.cluster_id,
            "cited_by_count": cand.cite_count,
        },
        "metadata": {
            "recruited_via": "courtlistener_v4",
            "recruited_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "filter": _CELL_DESCRIPTIONS[cell],
            "absolute_url": cand.absolute_url,
        },
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True, choices=list(_CELL_DESCRIPTIONS.keys()))
    parser.add_argument("--n", type=int, default=250)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cands = await _do_recruit(args.cell, args.n)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, cand in enumerate(cands):
            fh.write(json.dumps(_record(cand, args.cell, i), ensure_ascii=False) + "\n")
            n_written += 1
    print(f"Wrote {n_written} legal candidates to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
