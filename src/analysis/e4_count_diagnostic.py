"""POST-HOC diagnostic on E4: what survives EXACT citation-count equality?

NOT a registered estimand and not a substitute for one. The registered E4 result
lives in `data/analysis/phase_c/e4_naive_judge.json` and was produced by
`run_e4.py`, which is deliberately left untouched by this module so that the
one-shot artifact and the code that produced it stay in sync.

Two things the registered blocks cannot resolve on their own.

1. THE PRE-REGISTERED COUNT-MATCHED WINDOW IS ±1, AND ±1 IS PERMISSIVE HERE.
   Base checkpoints average 0.09 citations and the final checkpoints 1.31, so a
   ±1 window still admits the 0-against-1 contrast, which is the dominant
   comparison in the data. The registered control therefore removes much less of
   the citation-count confound than its name suggests.

2. THE PRE-REGISTERED PLACEBO IS NOT DIRECTION-FREE AS SPECIFIED. §5 assumes two
   post-trained outputs carry no expected direction, but SFT and DPO differ in
   STAGE, and citation surface keeps rising across the cascade (SFT 0.758
   citations, DPO 1.329). So a placebo above 50% is ambiguous between "the judge
   has a spurious directional bias" and "the judge is correctly reading a real
   increase", which are opposite conclusions.

Restricting to `count_delta == 0` resolves both: with citation counts identical
on both sides, any residual preference must come from the other three features,
and a placebo that returns to 50% shows the judge carries no bias of its own.

Only the logistic is refit. It is the registered primary model and it is
deterministic, so refitting reproduces it exactly rather than approximating it.
The DeBERTa arm cannot be included because `run_e4` did not persist its weights
and retraining would create a second version of a one-shot number.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from src.analysis.e4_naive_judge import bootstrap_ci, pair_outcome, preference_summary, training_rows
from src.analysis.e4_pairs import PLACEBO, PRIMARY, Pair, build_pairs, load_registry
from src.analysis.run_e4 import _logistic_scores, _read_jsonl, fit_logistic

EXACT = "exact_count_match"
TIERS = ("all", "within_one", EXACT)


def select_tier(pairs: Sequence[Pair], tier: str) -> list[Pair]:
    if tier == "all":
        return list(pairs)
    if tier == "within_one":
        return [p for p in pairs if p.is_count_matched]
    if tier == EXACT:
        return [p for p in pairs if p.count_delta == 0]
    raise ValueError(f"unknown tier {tier!r}; expected one of {TIERS}")


def score_tier(model, pairs: Sequence[Pair]) -> dict[str, Any]:
    if not pairs:
        return {"n_pairs": 0, "computable": False}
    outcomes = [
        pair_outcome(low, high)
        for low, high in zip(_logistic_scores(model, pairs, "low"), _logistic_scores(model, pairs, "high"))
    ]
    summary = preference_summary(outcomes)
    ci_low, ci_high = bootstrap_ci(outcomes)
    summary.update({"ci_low": ci_low, "ci_high": ci_high, "computable": True})
    return summary


def run_diagnostic(
    extractions: Sequence[Mapping[str, Any]], registry: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    model = fit_logistic(training_rows(extractions, registry))
    result: dict[str, Any] = {
        "note": "POST-HOC diagnostic, not a registered estimand. Logistic arm only.",
        "model": "logistic",
        "tiers": {},
    }
    for kind in (PRIMARY, PLACEBO):
        pairs = [p for p in build_pairs(extractions, registry, kind=kind) if p.split == "eval"]
        result["tiers"][kind] = {tier: score_tier(model, select_tier(pairs, tier)) for tier in TIERS}
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact-count-matched diagnostic on the E4 logistic judge.")
    parser.add_argument("--extractions", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    result = run_diagnostic(_read_jsonl(args.extractions), load_registry(args.registry))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
