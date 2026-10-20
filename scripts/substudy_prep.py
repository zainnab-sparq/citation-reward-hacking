"""Prep the EXISTENT_WRONG_ID sub-study (#3) for Workflow dispatch.

Builds the deterministic stratified 200-cit verifier-REAL subsample, writes each
citation's byte-identical v2 §6 judge prompt to a dedicated cache dir, emits ONE
self-contained Workflow batch script (reusing the production dispatch template
verbatim: model='sonnet' + WebSearch + WebFetch), and writes a manifest
(allocation, population sizes, selected custom_ids, seed) to the repo for
reproducibility and later population-reweighted scoring.

Dispatch is via the Workflow tool (Max OAuth, $0) — NEVER claude -p.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.analysis.substudy_sample import (
    SUBSTUDY_FLOOR,
    SUBSTUDY_N,
    SUBSTUDY_SEED,
    select_substudy_sample,
)
from src.judge_via_subagents import _build_qmap, build_judge_inputs
from src.judge_via_workflow import JudgeWorkflowItem, make_workflow_args
from src.run_judge_workflow import _WORKFLOW_TEMPLATE

EXTR = Path("data/extractions/phase_b/extractions.jsonl")
VERIF = Path("data/verifications/phase_b/verifications.jsonl")
QMAP = Path("data/questions/frozen_900q_legal_amended.jsonl")

CACHE = Path.home() / ".cache" / "citation-substudy-prompts"
PROMPTS_DIR = CACHE / "prompts"
BATCHES_DIR = CACHE / "batches"
MANIFEST_DIR = Path("data/judgments/phase_b_substudy")


def main() -> int:
    qmap = _build_qmap(QMAP)
    population_inputs = list(
        build_judge_inputs(EXTR, VERIF, QMAP, fire_on_statuses=("REAL",))
    )
    sample = select_substudy_sample(
        population_inputs, qmap, n=SUBSTUDY_N, floor=SUBSTUDY_FLOOR, seed=SUBSTUDY_SEED
    )

    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    items: list[JudgeWorkflowItem] = []
    for inp in sample.selected:
        path = PROMPTS_DIR / f"{inp.custom_id}.txt"
        path.write_text(inp.prompt, encoding="utf-8")
        items.append(JudgeWorkflowItem(custom_id=inp.custom_id, prompt_path=str(path)))

    name = "j2a-substudy-existent-wrong-id"
    blob = make_workflow_args(items)
    script = _WORKFLOW_TEMPLATE.format(
        name=name,
        name_json=json.dumps(name),
        items_json=json.dumps(blob["items"]),
    )
    batch_path = BATCHES_DIR / "substudy_batch.js"
    batch_path.write_text(script, encoding="utf-8")

    manifest = {
        "study": "EXISTENT_WRONG_ID verifier-REAL trust sub-study (amendment #6 #3)",
        "seed": sample.seed,
        "n_requested": sample.n_requested,
        "n_selected": len(sample.selected),
        "population_total": len(population_inputs),
        "strata": [
            {
                "domain": d,
                "cell": c,
                "population": sample.population[(d, c)],
                "allocation": sample.allocation.get((d, c), 0),
            }
            for (d, c) in sorted(sample.population)
        ],
        "selected_custom_ids": [i.custom_id for i in sample.selected],
        "batch_script": str(batch_path),
        "output_jsonl": str(MANIFEST_DIR / "judgments.jsonl"),
    }
    manifest_path = MANIFEST_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Console summary.
    print(f"population (verifier-REAL): {len(population_inputs)}")
    print(f"selected: {len(sample.selected)}  (seed={sample.seed}, floor={SUBSTUDY_FLOOR})\n")
    print(f"{'domain':10s} {'cell':14s} {'pop':>5s} {'draw':>5s} {'pop%':>6s}")
    tot = len(population_inputs)
    for (d, c) in sorted(sample.population):
        N = sample.population[(d, c)]
        a = sample.allocation.get((d, c), 0)
        print(f"{d:10s} {c:14s} {N:5d} {a:5d} {N/tot:6.1%}")
    print(f"\nbatch script: {batch_path}")
    print(f"manifest:     {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
