"""Run the two-step negative-existence proof on unanswerable candidates.

Reads a JSONL of unanswerable candidates, runs Levenshtein + SPECTER cosine
against OpenAlex top-5 hits, splits into kept vs rejected files, logs full
proof results for each candidate.

Run:
    .venv/Scripts/python.exe -m src.dataset_construction.validate_unanswerable \\
        --input data/questions/raw/academic_unanswerable.jsonl \\
        --output data/questions/validated/academic_unanswerable_kept.jsonl \\
        --rejected data/questions/validated/academic_unanswerable_rejected.jsonl \\
        --log data/questions/unanswerable_validation.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.dataset_construction.negative_existence import (
    NegativeExistenceProver,
    ProofResult,
)
from src.dataset_construction.openalex_client import OpenAlexClient
from src.dataset_construction.specter_embedder import SpecterEmbedder


def _proof_to_dict(p: ProofResult) -> dict:
    return {
        "passes": p.passes,
        "rejection": (
            {
                "step": p.rejection.step,
                "matched_title": p.rejection.matched_title,
                "matched_score": p.rejection.matched_score,
            }
            if p.rejection
            else None
        ),
        "cosine_max": p.cosine_max,
        "cosine_check_skipped_reason": p.cosine_check_skipped_reason,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()

    mailto = os.environ.get("OPENALEX_MAILTO")
    if not mailto:
        raise SystemExit("set OPENALEX_MAILTO env var")

    oa = OpenAlexClient(mailto=mailto)
    embedder = SpecterEmbedder()
    _ = embedder.model  # force load
    prover = NegativeExistenceProver(openalex=oa, embedder=embedder, cosine_threshold=0.85)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.rejected.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_kept = 0
    n_rejected_lev = 0
    n_rejected_cos = 0
    n_no_hits = 0

    with args.input.open("r", encoding="utf-8") as fin, \
         args.output.open("w", encoding="utf-8") as fout, \
         args.rejected.open("w", encoding="utf-8") as frej, \
         args.log.open("a", encoding="utf-8") as flog:
        for line in fin:
            if not line.strip():
                continue
            n_in += 1
            record = json.loads(line)
            title = (record.get("expected_citation") or {}).get("title", "")
            if not title:
                continue
            try:
                result = await prover.prove(candidate_title=title)
            except Exception as exc:
                # Network / API failure — skip this candidate (don't reject;
                # we'd rather rerun later than mislabel)
                print(f"  [error] {record['id']}: {exc!r}", file=sys.stderr)
                continue

            log_entry = {
                "id": record["id"],
                "title": title,
                "result": _proof_to_dict(result),
                "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            }
            flog.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

            record.setdefault("metadata", {})["negative_existence_proof"] = _proof_to_dict(result)
            if result.passes:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_kept += 1
                if result.cosine_check_skipped_reason == "no_hits":
                    n_no_hits += 1
            else:
                frej.write(json.dumps(record, ensure_ascii=False) + "\n")
                if result.rejection.step == "levenshtein_fuzzy":
                    n_rejected_lev += 1
                elif result.rejection.step == "cosine_similarity":
                    n_rejected_cos += 1

    print(f"Validation summary for {args.input.name}:")
    print(f"  input candidates : {n_in}")
    print(f"  kept             : {n_kept}")
    print(f"    of which no OpenAlex hits at all (passes by default): {n_no_hits}")
    print(f"  rejected (lev3)  : {n_rejected_lev}")
    print(f"  rejected (cosine): {n_rejected_cos}")


if __name__ == "__main__":
    asyncio.run(main())
