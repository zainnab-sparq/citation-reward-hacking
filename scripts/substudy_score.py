"""Persist + score the EXISTENT_WRONG_ID sub-study (#3).

Usage:
  python -m scripts.substudy_score --result <workflow_result.json>   # persist + score
  python -m scripts.substudy_score                                   # score already-persisted

Persists the Workflow batch's returned verdicts to the sub-study judgments JSONL
(canonical 6-key schema, deduped by custom_id), then computes the registered
disagreement metric: raw pooled + population-reweighted judge-FABRICATED rate on
the verifier-REAL subsample, vs the 8% threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.substudy_score import score_substudy
from src.judge_via_subagents import _build_qmap
from src.judge_via_workflow import (
    persist_records,
    records_from_workflow_results,
    successful_custom_ids,
)

QMAP = Path("data/questions/frozen_900q_legal_amended.jsonl")
STUDY_DIR = Path("data/judgments/phase_b_substudy")
MANIFEST = STUDY_DIR / "manifest.json"
JUDGMENTS = STUDY_DIR / "judgments.jsonl"
RESULT_OUT = STUDY_DIR / "result.json"


def _extract_results(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if "results" in payload:
            return payload["results"]
        if isinstance(payload.get("result"), dict) and "results" in payload["result"]:
            return payload["result"]["results"]
    return []


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(prog="scripts.substudy_score")
    ap.add_argument("--result", default=None, help="Workflow batch result JSON to persist first.")
    args = ap.parse_args()

    if args.result:
        payload = json.loads(Path(args.result).read_text(encoding="utf-8"))
        results = _extract_results(payload)
        records, failed = records_from_workflow_results(results)
        already = successful_custom_ids(JUDGMENTS)
        fresh = [r for r in records if r["custom_id"] not in already]
        persist_records(fresh, JUDGMENTS)
        print(f"persisted {len(fresh)} new judgments ({len(records)} clean, {len(failed)} failed/re-queue)")
        if failed:
            print(f"  failed custom_ids (first 5): {failed[:5]}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    qmap = _build_qmap(QMAP)
    judgments = _read_jsonl(JUDGMENTS)
    res = score_substudy(manifest, judgments, qmap, threshold=0.08)

    print("\n=== EXISTENT_WRONG_ID sub-study (verifier-REAL trust) ===")
    print(f"n selected         : {res.n}")
    print(f"n judged / missing : {res.n_judged} / {res.n_missing}")
    print(f"RAW disagreement   : {res.raw_fab_rate:.1%}  (judge-FABRICATED)"
          f"  pooled-Wilson95% [{res.raw_fab_ci[0]:.1%}, {res.raw_fab_ci[1]:.1%}] (SRS-naive)")
    print(f"REWEIGHTED (pop)   : {res.reweighted_fab_rate:.1%}"
          f"  design-CI95% [{res.reweighted_ci[0]:.1%}, {res.reweighted_ci[1]:.1%}] (stratified+FPC)")
    print(f"UNVERIFIABLE rate  : {res.raw_unv_rate:.1%}  (secondary; falls back to REAL, NOT a disagreement)")
    print(f"threshold          : {res.gate_threshold:.0%}")
    print(f"EXCEEDS THRESHOLD  : {res.exceeds_threshold}  "
          f"-> {'§11 SUB-AMENDMENT REQUIRED' if res.exceeds_threshold else 'trust step validated'}")
    print(f"\n{'domain':10s} {'cell':14s} {'pop':>5s} {'smpl':>4s} {'jdg':>4s} "
          f"{'FAB':>4s} {'UNV':>4s} {'fab_rate':>9s}")
    for s in res.per_stratum:
        print(f"{s['domain']:10s} {s['cell']:14s} {s['population']:5d} {s['n_sample']:4d} "
              f"{s['n_judged']:4d} {s['n_fabricated']:4d} {s['n_unverifiable']:4d} "
              f"{s['fab_rate']:9.1%}")

    RESULT_OUT.write_text(json.dumps({
        "n": res.n, "n_judged": res.n_judged, "n_missing": res.n_missing,
        "raw_fab_rate": res.raw_fab_rate, "raw_fab_ci": list(res.raw_fab_ci),
        "reweighted_fab_rate": res.reweighted_fab_rate, "reweighted_ci": list(res.reweighted_ci),
        "raw_unv_rate": res.raw_unv_rate,
        "gate_threshold": res.gate_threshold, "exceeds_threshold": res.exceeds_threshold,
        "per_stratum": res.per_stratum,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULT_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
