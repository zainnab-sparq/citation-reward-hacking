"""Score the paired re-judge sub-study.

The sub-study re-judges already-judged citations under the current instrument,
reading the byte-identical original prompt. Because the ITEM is held fixed, a
verdict difference is attributable to the instrument rather than to which
citations happened to fall in which era. That is what the observational era
comparison cannot do, and it is the whole reason the sub-study exists.

Three things are computed:

  transition matrix    original verdict x re-judged verdict, per era. Shows the
                       DIRECTION of movement, which a net rate hides: a 20% flip
                       rate made of UNVERIFIABLE -> FABRICATED means something
                       very different from one made of FABRICATED -> REAL.
  flip rate            per condition, with a Wilson interval.
  differentiality      does the flip probability depend on the prompt condition?
                       If not, the misclassification is non-differential and the
                       registered contrasts are attenuated (conservative) rather
                       than biased. This is the question the whole drift
                       investigation reduces to.

Weighting note: the ACROSS-CONDITION comparison uses UNWEIGHTED per-condition
rates, because it compares strata to each other. The design weights are only for
recovering a CORPUS-level flip rate from the balanced sample, a different
estimand, reported separately with a design-based interval and FPC.

Specification: docs/drift_interaction_prespec.md §9.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from scipy.stats import chi2_contingency

from src.analysis.substudy_score import wilson_ci, _Z_95

STATUSES = ("REAL", "FABRICATED", "UNVERIFIABLE")


def _in_era(pairs: Iterable[dict[str, Any]], era: str | None) -> list[dict[str, Any]]:
    return [p for p in pairs if era is None or p["era"] == era]


def transition_matrix(
    pairs: Iterable[dict[str, Any]], era: str | None = None
) -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter()
    for pair in _in_era(pairs, era):
        counts[(pair["original_status"], pair["rejudged_status"])] += 1
    return dict(counts)


def flip_rate_by_condition(
    pairs: Iterable[dict[str, Any]], era: str | None = None
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in _in_era(pairs, era):
        grouped[pair["condition"]].append(pair)

    out: dict[str, dict[str, Any]] = {}
    for condition, group in grouped.items():
        n = len(group)
        n_flipped = sum(1 for p in group if p["original_status"] != p["rejudged_status"])
        out[condition] = {
            "n": n,
            "n_flipped": n_flipped,
            "flip_rate": n_flipped / n if n else 0.0,
            "ci": wilson_ci(n_flipped, n),
        }
    return out


def differentiality_test(
    pairs: Iterable[dict[str, Any]], era: str | None = None
) -> dict[str, Any]:
    """Chi-square test of independence between condition and flipping.

    A null result is the reassuring one, but it BOUNDS the evidence rather than
    proving non-differentiality, so the achieved n is returned alongside.
    """
    rates = flip_rate_by_condition(pairs, era)
    conditions = sorted(rates)
    if len(conditions) < 2:
        return {"p_value": None, "differential": None, "n": sum(r["n"] for r in rates.values()),
                "reason": "fewer than two conditions present"}
    table = [
        [rates[c]["n_flipped"], rates[c]["n"] - rates[c]["n_flipped"]] for c in conditions
    ]
    # A column of structural zeros (nothing flipped anywhere, or everything did)
    # makes the test undefined rather than significant.
    if all(row[0] == 0 for row in table) or all(row[1] == 0 for row in table):
        return {"p_value": None, "differential": None,
                "n": sum(r["n"] for r in rates.values()),
                "reason": "no variation in flipping"}
    chi2, p, df, _ = chi2_contingency(table)
    return {
        "chi2": float(chi2),
        "p_value": float(p),
        "df": int(df),
        "n": sum(r["n"] for r in rates.values()),
        "conditions": conditions,
        "differential": bool(p < 0.05),
    }


def reweighted_flip_rate(
    pairs: Iterable[dict[str, Any]], era: str | None = None
) -> dict[str, Any]:
    """Corpus-level flip rate recovered from the balanced sample.

    Sum over strata of (N_h / N) * r_h, with a design-based SE carrying the
    finite-population correction, matching the Phase B sub-study's machinery.

    N_h is read from the frozen manifest rather than reconstructed as
    design_weight * n_observed, so a stratum that loses items to API failures
    keeps its correct weight instead of silently shrinking.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in _in_era(pairs, era):
        grouped[pair["condition"]].append(pair)

    strata = []
    for condition, group in grouped.items():
        n = len(group)
        if not n:
            continue
        population = group[0]["stratum_population"]
        strata.append({
            "condition": condition,
            "n": n,
            "population": population,
            "n_flipped": sum(1 for p in group if p["original_status"] != p["rejudged_status"]),
        })
    pop_total = sum(s["population"] for s in strata)
    if not pop_total:
        return {"rate": 0.0, "ci": (0.0, 0.0), "per_stratum": strata}

    rate = 0.0
    variance = 0.0
    for stratum in strata:
        r_h = stratum["n_flipped"] / stratum["n"]
        w_h = stratum["population"] / pop_total
        fpc = 1 - (stratum["n"] / stratum["population"]) if stratum["population"] else 1.0
        rate += w_h * r_h
        variance += w_h * w_h * r_h * (1 - r_h) / stratum["n"] * max(fpc, 0.0)
        stratum["flip_rate"] = r_h
    se = variance ** 0.5
    return {
        "rate": rate,
        "se": se,
        "ci": (max(0.0, rate - _Z_95 * se), min(1.0, rate + _Z_95 * se)),
        "per_stratum": sorted(strata, key=lambda s: s["condition"]),
    }


def score_rejudge(
    manifest: Sequence[dict[str, Any]], rejudgments: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Join the frozen sample manifest to the re-judgments and summarise."""
    verdicts = {
        r["custom_id"]: r.get("judge_status")
        for r in rejudgments
        if r.get("error") is None and r.get("judge_status")
    }

    pairs: list[dict[str, Any]] = []
    for item in manifest:
        rejudged = verdicts.get(item["custom_id"])
        if rejudged is None:
            continue
        pairs.append({**item, "rejudged_status": rejudged})

    by_era: dict[str, Any] = {}
    for era in sorted({item["era"] for item in manifest}):
        era_pairs = _in_era(pairs, era)
        by_era[era] = {
            "n": len(era_pairs),
            "transition_matrix": {
                f"{a}->{b}": n for (a, b), n in sorted(transition_matrix(era_pairs).items())
            },
            "flip_rate_by_condition": flip_rate_by_condition(era_pairs),
            "differentiality": differentiality_test(era_pairs),
            "reweighted_flip_rate": reweighted_flip_rate(era_pairs),
        }

    return {
        "n_sampled": len(manifest),
        "n_rejudged": len(pairs),
        "n_missing": len(manifest) - len(pairs),
        "by_era": by_era,
        "note": (
            "Across-condition comparisons use UNWEIGHTED per-condition rates "
            "(strata compared to each other). Design weights apply only to the "
            "corpus-level reweighted rate. A null differentiality test bounds the "
            "evidence; it does not prove non-differentiality."
        ),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score the paired re-judge sub-study.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--rejudgments", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    result = score_rejudge(_read_jsonl(args.manifest), _read_jsonl(args.rejudgments))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"sampled {result['n_sampled']}, re-judged {result['n_rejudged']}, "
          f"missing {result['n_missing']}")
    for era, block in result["by_era"].items():
        d = block["differentiality"]
        print(f"\n[{era}] n={block['n']}")
        for transition, n in block["transition_matrix"].items():
            print(f"    {transition:<32} {n}")
        for condition, row in sorted(block["flip_rate_by_condition"].items()):
            print(f"    flip {condition:<22} {row['n_flipped']}/{row['n']} "
                  f"= {row['flip_rate']:.3f}")
        if d.get("p_value") is None:
            print(f"    differentiality: not testable ({d.get('reason')})")
        else:
            print(f"    differentiality: chi2={d['chi2']:.3f} df={d['df']} "
                  f"p={d['p_value']:.4g} differential={d['differential']}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
