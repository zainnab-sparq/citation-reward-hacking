"""Phase C extraction COST PILOT (throwaway; not a research artifact).

Submits a stratified 20-per-file (=240) sample of the Phase C generations
through the REAL extraction path (src.extract_batch.build_extraction_requests,
Haiku 4.5 extractor, batch API), sums actual usage tokens from the batch
results, and projects the full 43,200-record extraction cost at Haiku batch
rates. Purpose: ground-truth the cap before the full spend ("pilot -> read
usage -> cap -> Batch"). Writes no extraction artifact.
"""
from __future__ import annotations

import asyncio
import glob
import json
import os
from pathlib import Path

from anthropic import AsyncAnthropic

from src.extract_batch import (
    build_extraction_requests,
    build_question_domain_map_from_path,
)

_HAIKU = "claude-haiku-4-5-20251001"
_RATE_IN_BATCH = 0.50 / 1_000_000   # $1/MTok input, -50% batch
_RATE_OUT_BATCH = 2.50 / 1_000_000  # $5/MTok output, -50% batch
_QUESTIONS = Path("data/questions/frozen_900q_legal_amended.jsonl")
_PER_FILE = 20
_FULL_N = 43_200


def _stratified_sample() -> list[dict]:
    rows: list[dict] = []
    for f in sorted(glob.glob("data/generations/phase_c/*.jsonl")):
        lines = [l for l in Path(f).read_text(encoding="utf-8").splitlines() if l.strip()]
        step = max(1, len(lines) // _PER_FILE)
        for l in lines[::step][:_PER_FILE]:
            rows.append(json.loads(l))
    return rows


async def main() -> None:
    gens = _stratified_sample()
    qdomain = build_question_domain_map_from_path(_QUESTIONS)
    requests = build_extraction_requests(
        generations=gens,
        question_domain=qdomain,
        extractor_model_id=_HAIKU,
        max_output_tokens=2048,
    )
    n_empty = sum(1 for g in gens if not (g.get("output_text") or "").strip())
    print(f"pilot sample: {len(gens)} gens ({n_empty} empty) -> {len(requests)} requests")

    client = AsyncAnthropic()
    resp = await client.messages.batches.create(requests=requests)
    batch_id = resp.id
    print(f"submitted batch {batch_id}")

    while True:
        status = await client.messages.batches.retrieve(batch_id)
        if status.processing_status == "ended":
            break
        await asyncio.sleep(15)

    tin = tout = n_ok = n_err = 0
    iterator = await client.messages.batches.results(batch_id)
    async for item in iterator:
        r = item.result
        if getattr(r, "type", None) == "succeeded":
            u = r.message.usage
            tin += u.input_tokens
            tout += u.output_tokens
            n_ok += 1
        else:
            n_err += 1

    cost = tin * _RATE_IN_BATCH + tout * _RATE_OUT_BATCH
    per = cost / max(n_ok, 1)
    print(f"RESULT ok={n_ok} err={n_err} in_tok={tin:,} out_tok={tout:,}")
    print(f"pilot cost=${cost:.4f}  per_request=${per:.6f}")
    print(f"PROJECTED full {_FULL_N:,}-request extraction = ${per * _FULL_N:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
