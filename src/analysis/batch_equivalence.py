"""Batch-equivalence check for the J2a judge layer (pre-reg §11 deviation #9).

The J2a judge ran on two execution paths:
  - paid `claude -p` subprocess  (n=3,821, halted by the billing leak)
  - free Workflow-tool `agent()`  (n=6,872, the remainder)

§11 #9 asks whether the free path produced judge verdicts equivalent to the
paid path. Both ran byte-identical v2 §6 prompts on Sonnet 4.6 with WebSearch +
WebFetch, so the verdict distribution should not depend on the dispatch
primitive. This module measures that, controlling for the cell composition each
path happened to cover.

Provenance is recovered from the pre-pivot backup
(`*.before_repurge_*.bak`): every citation the subprocess failed to judge was
left as an `error: subprocess_failed_or_unparseable` placeholder, and those are
exactly the citations the Workflow path judged afterward. So a citation is
`subprocess` iff its backup row is a real verdict, and `workflow` iff its backup
row is the placeholder. This is content-level provenance, not row-order
inference.

IMPORTANT (disclosed limitation): path assignment is NOT randomized. The
subprocess judged whatever cleared before the halt, so within-stratum residual
confounding (which specific citations each arm got) is possible. This is a
sensitivity check, not a randomized equivalence trial.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any

from scipy.stats import chi2_contingency

from src.analysis.aggregate import parse_judgment_custom_id

PLACEHOLDER_MARKER = "subprocess_failed_or_unparseable"
JUDGE_STATUSES = ("REAL", "FABRICATED", "UNVERIFIABLE")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rows.append(json.loads(raw))
    return rows


def partition_provenance(
    backup_records: Iterable[dict[str, Any]],
    *,
    placeholder_marker: str = PLACEHOLDER_MARKER,
) -> dict[str, str]:
    """Map each custom_id to "subprocess" or "workflow" from the backup.

    A backup row carrying the placeholder error was left unjudged by the
    subprocess and later judged by the Workflow path; anything else was
    completed by the subprocess.
    """
    prov: dict[str, str] = {}
    for record in backup_records:
        cid = record["custom_id"]
        prov[cid] = "workflow" if record.get("error") == placeholder_marker else "subprocess"
    return prov


def build_equivalence_rows(
    judgments: Iterable[dict[str, Any]],
    provenance: dict[str, str],
    questions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach provenance + (condition, cell, domain) to each current judgment.

    Judgments with no provenance entry (cannot be attributed to a path) are
    dropped; the caller can recover the count by comparing lengths.
    """
    cell_domain = {q["id"]: (q["cell"], q["domain"]) for q in questions}
    rows: list[dict[str, Any]] = []
    for j in judgments:
        cid = j["custom_id"]
        prov = provenance.get(cid)
        if prov is None:
            continue
        qid, _model, condition, _trial, _cidx = parse_judgment_custom_id(cid)
        cd = cell_domain.get(qid)
        if cd is None:
            continue
        cell, domain = cd
        rows.append({
            "custom_id": cid,
            "provenance": prov,
            "condition": condition,
            "cell": cell,
            "domain": domain,
            "judge_status": j.get("judge_status"),
            "judge_search_count": j.get("judge_search_count"),
        })
    return rows


def _valid_status(value: Any) -> bool:
    return value in JUDGE_STATUSES


def overall_chi_square(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Chi-square test of judge_status independence from provenance.

    Rows whose judge_status is not one of the three real verdicts (e.g. None /
    error) are excluded and counted in `n_excluded`.
    """
    rows = list(rows)
    valid = [r for r in rows if _valid_status(r["judge_status"])]
    n_excluded = len(rows) - len(valid)

    provs = ("subprocess", "workflow")
    table = [
        [sum(1 for r in valid if r["provenance"] == p and r["judge_status"] == s)
         for s in JUDGE_STATUSES]
        for p in provs
    ]
    # Drop all-zero status columns so chi2 does not divide by a zero margin.
    nonzero_cols = [j for j in range(len(JUDGE_STATUSES))
                    if any(table[i][j] > 0 for i in range(len(provs)))]
    reduced = [[table[i][j] for j in nonzero_cols] for i in range(len(provs))]

    chi2, p, dof, _expected = chi2_contingency(reduced)
    return {
        "chi2": float(chi2),
        "p": float(p),
        "dof": int(dof),
        "n": len(valid),
        "n_excluded": n_excluded,
        "statuses": [JUDGE_STATUSES[j] for j in nonzero_cols],
        "table": {provs[i]: dict(zip(JUDGE_STATUSES, table[i])) for i in range(len(provs))},
    }


def _proportions(rows: list[dict[str, Any]]) -> dict[str, float]:
    valid = [r["judge_status"] for r in rows if _valid_status(r["judge_status"])]
    counts = Counter(valid)
    n = len(valid)
    return {s: (counts.get(s, 0) / n if n else 0.0) for s in JUDGE_STATUSES}


def per_stratum_deltas(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per (condition × cell × domain): subprocess vs workflow verdict
    proportions and their absolute gaps. A stratum present in only one arm is
    flagged `insufficient` rather than given a misleading delta.
    """
    strata: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
    for r in rows:
        key = (r["condition"], r["cell"], r["domain"])
        arm = strata.setdefault(key, {"subprocess": [], "workflow": []})
        arm[r["provenance"]].append(r)

    out: list[dict[str, Any]] = []
    for (condition, cell, domain), arm in sorted(strata.items()):
        sub_rows, wf_rows = arm["subprocess"], arm["workflow"]
        n_sub, n_wf = len(sub_rows), len(wf_rows)
        prop_sub, prop_wf = _proportions(sub_rows), _proportions(wf_rows)
        insufficient = n_sub == 0 or n_wf == 0
        deltas = {s: abs(prop_sub[s] - prop_wf[s]) for s in JUDGE_STATUSES}
        out.append({
            "condition": condition,
            "cell": cell,
            "domain": domain,
            "n_subprocess": n_sub,
            "n_workflow": n_wf,
            "prop_subprocess": prop_sub,
            "prop_workflow": prop_wf,
            "abs_deltas": deltas,
            "max_abs_delta": None if insufficient else max(deltas.values()),
            "insufficient": insufficient,
        })
    return out


def search_count_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Mean/median judge_search_count by arm, overall and per stratum.

    Difficulty-selection diagnostic: the transient-skip re-queue pushed every
    subprocess failure into the workflow batch, so if workflow-judged citations
    needed systematically more searches, the verdict-distribution gap is
    explained by citation difficulty, not the dispatch primitive.
    """
    provs = ("subprocess", "workflow")
    valid: list[dict[str, Any]] = []
    n_skipped = 0
    for r in rows:
        if isinstance(r.get("judge_search_count"), int):
            valid.append(r)
        else:
            n_skipped += 1

    def _arm_stats(arm_rows: list[dict[str, Any]]) -> dict[str, Any]:
        counts = [r["judge_search_count"] for r in arm_rows]
        if not counts:
            return {"n": 0, "mean": None, "median": None}
        return {"n": len(counts), "mean": float(mean(counts)),
                "median": float(median(counts))}

    overall = {p: _arm_stats([r for r in valid if r["provenance"] == p]) for p in provs}
    overall["delta_mean"] = (
        overall["workflow"]["mean"] - overall["subprocess"]["mean"]
        if overall["workflow"]["mean"] is not None
        and overall["subprocess"]["mean"] is not None
        else None
    )

    strata: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
    for r in valid:
        key = (r["condition"], r["cell"], r["domain"])
        arm = strata.setdefault(key, {"subprocess": [], "workflow": []})
        arm[r["provenance"]].append(r)

    per_stratum: list[dict[str, Any]] = []
    for (condition, cell, domain), arm in sorted(strata.items()):
        sub, wf = _arm_stats(arm["subprocess"]), _arm_stats(arm["workflow"])
        per_stratum.append({
            "condition": condition,
            "cell": cell,
            "domain": domain,
            "n_subprocess": sub["n"],
            "n_workflow": wf["n"],
            "mean_subprocess": sub["mean"],
            "mean_workflow": wf["mean"],
            "delta_mean": (wf["mean"] - sub["mean"]
                           if sub["mean"] is not None and wf["mean"] is not None
                           else None),
        })
    return {"overall": overall, "per_stratum": per_stratum,
            "n_skipped_no_count": n_skipped}


def gap_summary(
    strata: Iterable[dict[str, Any]],
    *,
    min_arm_n: int = 30,
) -> dict[str, Any]:
    """Summaries that keep sparse strata from driving the headline gap.

    The raw max_abs_delta over all strata is dominated by tiny cells (a 28.6pp
    "gap" at n=11 is noise); the n-weighted mean and the well-powered subset
    (both arms >= min_arm_n) are the reportable numbers.
    """
    comparable = [s for s in strata if not s["insufficient"]]
    if not comparable:
        return {"weighted_mean_max_gap": None, "min_arm_n": min_arm_n,
                "well_powered": {"n_strata": 0, "mean_max_gap": None,
                                 "max_gap": None, "max_gap_stratum": None}}

    weights = [s["n_subprocess"] + s["n_workflow"] for s in comparable]
    weighted_mean = (
        sum(w * s["max_abs_delta"] for w, s in zip(weights, comparable)) / sum(weights)
    )

    powered = [s for s in comparable
               if min(s["n_subprocess"], s["n_workflow"]) >= min_arm_n]
    if powered:
        worst = max(powered, key=lambda s: s["max_abs_delta"])
        well_powered = {
            "n_strata": len(powered),
            "mean_max_gap": float(mean(s["max_abs_delta"] for s in powered)),
            "max_gap": worst["max_abs_delta"],
            "max_gap_stratum": {"condition": worst["condition"],
                                "cell": worst["cell"],
                                "domain": worst["domain"]},
        }
    else:
        well_powered = {"n_strata": 0, "mean_max_gap": None,
                        "max_gap": None, "max_gap_stratum": None}

    return {"weighted_mean_max_gap": weighted_mean, "min_arm_n": min_arm_n,
            "well_powered": well_powered}


def run(
    *,
    judgments_path: Path,
    backup_path: Path,
    questions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Load artifacts, compute the equivalence report, write + return it."""
    judgments = _read_jsonl(judgments_path)
    backup = _read_jsonl(backup_path)
    questions = _read_jsonl(questions_path)
    questions = [{"id": q["id"], "cell": q["cell"], "domain": q["domain"]} for q in questions]

    provenance = partition_provenance(backup)
    rows = build_equivalence_rows(judgments, provenance, questions)

    n_sub = sum(1 for r in rows if r["provenance"] == "subprocess")
    n_wf = sum(1 for r in rows if r["provenance"] == "workflow")
    overall = overall_chi_square(rows)
    strata = per_stratum_deltas(rows)
    comparable = [s for s in strata if not s["insufficient"]]
    max_delta = max((s["max_abs_delta"] for s in comparable), default=None)

    report = {
        "n_judgments": len(judgments),
        "n_attributed": len(rows),
        "n_subprocess": n_sub,
        "n_workflow": n_wf,
        "overall_chi_square": overall,
        "n_strata": len(strata),
        "n_strata_comparable": len(comparable),
        "max_abs_delta_across_strata": max_delta,
        "gap_summary": gap_summary(strata),
        "search_count": search_count_summary(rows),
        "per_stratum": strata,
        "caveat": (
            "Path assignment is not randomized (subprocess judged whatever "
            "cleared before the billing halt); within-stratum residual "
            "confounding is possible. Sensitivity check, not a randomized "
            "equivalence trial."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.analysis.batch_equivalence",
        description="J2a subprocess-vs-workflow judge-verdict equivalence check.",
    )
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True,
                        help="Pre-pivot *.before_repurge_*.bak (provenance ground truth).")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    report = run(
        judgments_path=args.judgments,
        backup_path=args.backup,
        questions_path=args.questions,
        output_path=args.output,
    )
    print(f"attributed {report['n_attributed']} judgments "
          f"({report['n_subprocess']} subprocess / {report['n_workflow']} workflow)")
    oc = report["overall_chi_square"]
    print(f"overall chi-square: chi2={oc['chi2']:.2f} dof={oc['dof']} p={oc['p']:.4g}")
    print(f"max abs proportion delta across {report['n_strata_comparable']} "
          f"comparable strata: {report['max_abs_delta_across_strata']}")


if __name__ == "__main__":
    main()
