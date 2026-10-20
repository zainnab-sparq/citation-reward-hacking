"""Score the fabrication-enriched blind hand-check of the J2a pipeline.

The sample deliberately oversamples fabricated citations, so two kinds of number
come out of it and they must not be mixed:

  * POPULATION-level numbers (overall accuracy, the trivial-baseline
    comparison) are DESIGN WEIGHTED. Unweighted they would inherit the enriched
    sample's class balance rather than the corpus's, and the whole point of the
    enrichment is that those differ.
  * PER-CLASS and PER-STRATUM diagnostics are reported unweighted, because they
    are conditional on the class or stratum and the enrichment is what buys
    their precision. Weighted recall is also reported, since a weighted recall
    answers "what fraction of the corpus's fabrications does the pipeline
    catch", which is a different and equally fair question.

The headline comparison is the one Step 0 could not make: is the pipeline
distinguishable from a judge that answers REAL to everything? On the Step 0 set
it was not (88.0% vs 96.0%, McNemar p = 0.125).

Usage:
  python -m src.analysis.handcheck_score \
      --blind data/handcheck/phase_c/phase_c_blind_handcheck.csv \
      --reveal data/handcheck/phase_c/phase_c_labels_reveal.csv \
      --output data/analysis/phase_c/handcheck_result.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from scipy.stats import binomtest

from src.analysis.substudy_score import wilson_ci

VALID_LABELS = ("REAL", "FABRICATED", "UNVERIFIABLE")
_SHORTHAND = {"FAB": "FABRICATED", "UNV": "UNVERIFIABLE", "REAL_CONFIRMED": "REAL"}


def canon_label(raw: str) -> str | None:
    """Normalise an annotator's entry. Blank means not yet labelled."""
    text = (raw or "").strip().upper()
    if not text:
        return None
    text = _SHORTHAND.get(text, text)
    if text not in VALID_LABELS:
        raise ValueError(
            f"unrecognised hand-check label {raw!r}; expected one of "
            f"{', '.join(VALID_LABELS)} (or FAB / UNV)")
    return text


def _correct(row: dict[str, Any]) -> bool:
    return row["human_status"] == row["combined_status"]


def weighted_accuracy(rows: Sequence[dict[str, Any]]) -> float:
    total = sum(r["design_weight"] for r in rows)
    if not total:
        return 0.0
    return sum(r["design_weight"] for r in rows if _correct(r)) / total


def class_performance(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    support = [r for r in rows if r["human_status"] == label]
    predicted = [r for r in rows if r["combined_status"] == label]
    hit = [r for r in support if _correct(r)]
    true_positive = [r for r in predicted if _correct(r)]

    out: dict[str, Any] = {
        "label": label,
        "support": len(support),
        "n_predicted": len(predicted),
        "recall": len(hit) / len(support) if support else None,
        "recall_ci": wilson_ci(len(hit), len(support)) if support else None,
        "precision": len(true_positive) / len(predicted) if predicted else None,
        "precision_ci": wilson_ci(len(true_positive), len(predicted)) if predicted else None,
    }
    support_weight = sum(r["design_weight"] for r in support)
    out["weighted_recall"] = (
        sum(r["design_weight"] for r in hit) / support_weight if support_weight else None
    )
    return out


def baseline_comparison(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pipeline versus always answering the most common label.

    Both sides are weighted, because the trivial baseline's score is determined
    by the POPULATION class balance and the sample's balance is engineered.
    """
    weight_by_label: dict[str, float] = defaultdict(float)
    for row in rows:
        weight_by_label[row["human_status"]] += row["design_weight"]
    if not weight_by_label:
        return {"baseline_label": None, "mcnemar_p": None}
    baseline_label = max(weight_by_label, key=lambda k: weight_by_label[k])
    total = sum(weight_by_label.values())

    judge_right = sum(
        1 for r in rows if _correct(r) and r["human_status"] != baseline_label)
    judge_wrong = sum(
        1 for r in rows if not _correct(r) and r["human_status"] == baseline_label)
    discordant = judge_right + judge_wrong

    return {
        "baseline_label": baseline_label,
        "baseline_weighted_accuracy": weight_by_label[baseline_label] / total,
        "pipeline_weighted_accuracy": weighted_accuracy(rows),
        "judge_right_baseline_wrong": judge_right,
        "judge_wrong_baseline_right": judge_wrong,
        "mcnemar_p": (
            binomtest(judge_right, discordant, 0.5).pvalue if discordant else None),
    }


def _grouped_accuracy(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key) or "(none)"].append(row)
    out = {}
    for name, group in sorted(groups.items()):
        correct = sum(1 for r in group if _correct(r))
        out[name] = {
            "n": len(group),
            "n_correct": correct,
            "accuracy": correct / len(group),
            "accuracy_ci": wilson_ci(correct, len(group)),
        }
    return out


def by_stratum(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return _grouped_accuracy(rows, "stratum")


def by_era(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy per instrument regime.

    The drift analysis measures the judge's STABILITY across regimes. This is
    the only direct evidence on its ACCURACY within each, which is a different
    property: an instrument can be perfectly stable and consistently wrong.
    Cell counts are small, so read the intervals, not the point estimates.
    """
    return _grouped_accuracy(rows, "era")


def score_handcheck(
    blind: Sequence[dict[str, Any]], reveal: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    labels = {r["row_id"]: r for r in reveal}
    rows: list[dict[str, Any]] = []
    for entry in blind:
        human = canon_label(entry.get("your_status", ""))
        if human is None:
            continue
        meta = labels[entry["row_id"]]
        rows.append({
            "row_id": entry["row_id"],
            "human_status": human,
            "combined_status": meta["combined_status"],
            "stratum": meta["stratum"],
            "design_weight": float(meta["design_weight"]),
            "era": meta.get("era", ""),
        })

    correct = sum(1 for r in rows if _correct(r))
    return {
        "n_sampled": len(blind),
        "n_labelled": len(rows),
        "n_unlabelled": len(blind) - len(rows),
        "overall": {
            "unweighted_accuracy": correct / len(rows) if rows else 0.0,
            "unweighted_ci": wilson_ci(correct, len(rows)) if rows else None,
            "weighted_accuracy": weighted_accuracy(rows),
        },
        "baseline": baseline_comparison(rows),
        "per_class": {
            label: class_performance(rows, label) for label in VALID_LABELS
        },
        "by_stratum": by_stratum(rows),
        "by_era": by_era(rows),
        "note": (
            "Population-level numbers are design weighted; per-class and "
            "per-stratum numbers are unweighted and conditional. The sample is "
            "fabrication-enriched, so unweighted overall accuracy is NOT a "
            "corpus estimate."
        ),
    }


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the blind hand-check.")
    parser.add_argument("--blind", required=True, type=Path)
    parser.add_argument("--reveal", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    result = score_handcheck(_read_csv(args.blind), _read_csv(args.reveal))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"labelled {result['n_labelled']} of {result['n_sampled']} "
          f"({result['n_unlabelled']} still blank)")
    if not result["n_labelled"]:
        print("nothing to score yet.")
        return 0

    o, b = result["overall"], result["baseline"]
    lo, hi = o["unweighted_ci"]
    print(f"\nunweighted accuracy (enriched sample): {o['unweighted_accuracy']:.3f} "
          f"[{lo:.3f}, {hi:.3f}]")
    print(f"weighted accuracy (corpus estimate):   {o['weighted_accuracy']:.3f}")
    print(f"\ntrivial 'always {b['baseline_label']}' baseline (weighted): "
          f"{b['baseline_weighted_accuracy']:.3f}")
    print(f"pipeline (weighted):                        "
          f"{b['pipeline_weighted_accuracy']:.3f}")
    if b["mcnemar_p"] is not None:
        print(f"McNemar: {b['judge_right_baseline_wrong']} vs "
              f"{b['judge_wrong_baseline_right']} discordant, p = {b['mcnemar_p']:.4g}")

    fab = result["per_class"]["FABRICATED"]
    if fab["support"]:
        rlo, rhi = fab["recall_ci"]
        print(f"\nFABRICATED recall: {fab['recall']:.3f} [{rlo:.3f}, {rhi:.3f}] "
              f"on {fab['support']} human-labelled fabrications")
        if fab["precision"] is not None:
            plo, phi = fab["precision_ci"]
            print(f"FABRICATED precision: {fab['precision']:.3f} [{plo:.3f}, {phi:.3f}]")

    print("\nby stratum:")
    for name, row in result["by_stratum"].items():
        print(f"   {name:<24} {row['n_correct']}/{row['n']} = {row['accuracy']:.3f}")
    print("\nby era:")
    for name, row in result["by_era"].items():
        print(f"   {name:<24} {row['n_correct']}/{row['n']} = {row['accuracy']:.3f}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
