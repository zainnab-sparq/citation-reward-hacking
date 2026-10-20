"""Run E4 once: fit both judges, score the held-out family, apply the registered rules.

🔒 GATED. This produces a registered confirmatory estimand, so `--run-confirmatory`
is required and defaults to off, the same protection `phase_c_drift_interaction.R`
puts on its robustness panel. Running it is a one-shot act: the feature
definitions, the pairing, the tie rule, the bootstrap seed and every decision
threshold are fixed in `docs/e4_surface_feature_spec.md` beforehand, so there is
nothing left to iterate on and re-running with different settings after seeing a
rate would be a deviation.

E4 depends only on extraction records. It touches no verification and no judge
output, so it does not wait on the nightly judge loop and its result cannot move
when the judge finishes.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

from src.analysis.e4_naive_judge import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    bootstrap_ci,
    count_matched_verdict,
    evaluate_hypothesis,
    feature_matrix,
    pair_outcome,
    preference_summary,
    render_citation_text,
    select_primary_model,
    training_rows,
)
from src.analysis.e4_pairs import PLACEBO, PRIMARY, Pair, build_pairs, load_registry

LOGISTIC_SEED = 20260813
Scorer = Callable[[Sequence[Any]], Sequence[float]]


def fit_logistic(rows: Sequence[Mapping[str, Any]], seed: int = LOGISTIC_SEED):
    """The 4-feature hand-coded baseline, and E4's primary model unless DeBERTa
    beats it by more than 5pp. Features are standardised only so lbfgs converges;
    it is a monotone rescaling and changes no preference ordering."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    model.fit(feature_matrix(rows), [row["label"] for row in rows])
    return model


def _text_index(extractions: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], str]:
    return {
        (r["question_id"], r["model_id"], r["condition"]): render_citation_text(r.get("citations"))
        for r in extractions
        if not r.get("error")
    }


def _outcomes(pairs: Sequence[Pair], low_scores: Sequence[float], high_scores: Sequence[float]) -> list[float]:
    return [pair_outcome(low, high) for low, high in zip(low_scores, high_scores)]


def _score_block(pairs: Sequence[Pair], outcomes: Sequence[float], label: str) -> dict[str, Any]:
    summary = preference_summary(outcomes)
    ci_low, ci_high = bootstrap_ci(outcomes)
    summary.update({"ci_low": ci_low, "ci_high": ci_high, "block": label})
    return summary


def _logistic_scores(model, pairs: Sequence[Pair], side: str) -> list[float]:
    rows = [{"features": getattr(pair, f"{side}_features")} for pair in pairs]
    return [float(p) for p in model.predict_proba(feature_matrix(rows))[:, 1]]


def _deberta_scores(scorer: Scorer, pairs: Sequence[Pair], side: str, texts: Mapping[tuple[str, str, str], str]):
    keys = [(pair.question_id, getattr(pair, f"{side}_model_id"), pair.condition) for pair in pairs]
    return [float(s) for s in scorer([texts[key] for key in keys])]


def _evaluate_arm(name: str, low: Sequence[float], high: Sequence[float], pairs: Sequence[Pair]) -> dict[str, Any]:
    outcomes = _outcomes(pairs, low, high)
    unconstrained = _score_block(pairs, outcomes, "unconstrained")
    matched_pairs = [p for p in pairs if p.is_count_matched]
    matched_outcomes = [o for p, o in zip(pairs, outcomes) if p.is_count_matched]

    # An empty count-matched subsample is reported, not raised: the control
    # failing to exist is a result about the data, and a crash here would take
    # the unconstrained estimate down with it.
    if matched_outcomes:
        matched = _score_block(matched_pairs, matched_outcomes, "count_matched")
        matched["computable"] = True
        verdict = count_matched_verdict(matched["ci_low"], unconstrained["ci_low"])
    else:
        matched = {"block": "count_matched", "n_pairs": 0, "computable": False}
        verdict = "not_computable"

    return {
        "model": name,
        "unconstrained": unconstrained,
        "count_matched": matched,
        "count_matched_verdict": verdict,
        "hypothesis": evaluate_hypothesis(unconstrained["preference_rate"], unconstrained["ci_low"]),
    }


def run_e4(
    extractions: Sequence[Mapping[str, Any]],
    registry: Sequence[Mapping[str, Any]],
    *,
    deberta_fitter: Callable[[Sequence[str], Sequence[int]], Scorer] | None = None,
    log=print,
) -> dict[str, Any]:
    rows = training_rows(extractions, registry)
    log(f"training generations: {len(rows)} ({sum(r['label'] for r in rows)} post-trained)")

    eval_pairs = [p for p in build_pairs(extractions, registry, kind=PRIMARY) if p.split == "eval"]
    placebo_pairs = [p for p in build_pairs(extractions, registry, kind=PLACEBO) if p.split == "eval"]
    log(f"evaluation pairs: {len(eval_pairs)} primary, {len(placebo_pairs)} placebo")

    texts = _text_index(extractions)
    arms: dict[str, dict[str, Any]] = {}
    placebo: dict[str, Any] = {}

    log("fitting the 4-feature logistic ...")
    logistic = fit_logistic(rows)
    arms["logistic"] = _evaluate_arm(
        "logistic",
        _logistic_scores(logistic, eval_pairs, "low"),
        _logistic_scores(logistic, eval_pairs, "high"),
        eval_pairs,
    )
    placebo["logistic"] = _score_block(
        placebo_pairs,
        _outcomes(
            placebo_pairs,
            _logistic_scores(logistic, placebo_pairs, "low"),
            _logistic_scores(logistic, placebo_pairs, "high"),
        ),
        "placebo",
    )

    if deberta_fitter is not None:
        log("fine-tuning DeBERTa-v3-base ...")
        scorer = deberta_fitter([row["text"] for row in rows], [row["label"] for row in rows])
        arms["deberta"] = _evaluate_arm(
            "deberta",
            _deberta_scores(scorer, eval_pairs, "low", texts),
            _deberta_scores(scorer, eval_pairs, "high", texts),
            eval_pairs,
        )
        placebo["deberta"] = _score_block(
            placebo_pairs,
            _outcomes(
                placebo_pairs,
                _deberta_scores(scorer, placebo_pairs, "low", texts),
                _deberta_scores(scorer, placebo_pairs, "high", texts),
            ),
            "placebo",
        )

    result: dict[str, Any] = {
        "n_training_generations": len(rows),
        "n_eval_pairs": len(eval_pairs),
        "n_placebo_pairs": len(placebo_pairs),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "arms": arms,
        "placebo": placebo,
    }
    if "deberta" in arms:
        primary = select_primary_model(
            arms["logistic"]["unconstrained"]["preference_rate"],
            arms["deberta"]["unconstrained"]["preference_rate"],
        )
        result["primary_model"] = primary
        result["h1_4"] = arms[primary]["hypothesis"]
        result["h1_4_count_matched"] = arms[primary]["count_matched_verdict"]
    return result


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extractions", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--run-confirmatory",
        action="store_true",
        help="required: E4 is a one-shot registered test and refuses to run without it",
    )
    parser.add_argument("--skip-deberta", action="store_true", help="logistic arm only; no model is selected")
    args = parser.parse_args(argv)

    if not args.run_confirmatory:
        parser.error(
            "E4 produces a registered confirmatory estimand. Pass --run-confirmatory to run it, "
            "and read docs/e4_surface_feature_spec.md first: it runs once."
        )

    fitter = None
    if not args.skip_deberta:
        from src.analysis.e4_deberta import fit_deberta

        fitter = fit_deberta

    result = run_e4(_read_jsonl(args.extractions), load_registry(args.registry), deberta_fitter=fitter)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
