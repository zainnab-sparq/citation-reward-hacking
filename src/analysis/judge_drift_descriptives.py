"""Descriptive statistics for the J2a judge-drift subsection and figure F3.

Every number here comes from the judge's own append-only output stream, at the
citation level, with era read off row position exactly as `judge_era` defines
it. That source matters. The corrected verification file carries citations the
judge never saw, because the verifier-REAL trust step passed them straight
through, and a verdict mix computed over those would describe the pipeline
rather than the instrument.

Emits a single JSON artifact so the manuscript prose and the R figure code read
the same numbers rather than each deriving their own.

Usage (from repo root):
    python -m src.analysis.judge_drift_descriptives \\
        --judgments data/judgments/phase_c/claude-subagent-sonnet.jsonl \\
        --questions data/analysis_inputs/phase_c_final/questions.jsonl \\
        --models data/analysis_inputs/phase_c_final/models.jsonl \\
        --output data/analysis/phase_c/judge_drift_descriptives.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from src.analysis.judge_era import load_judgment_meta

ERAS = ("EARLY", "DEGRADED", "RECOVERED")
CELLS = ("known_real", "obscure_real", "unanswerable")
REAL_STATUS = "REAL"


def _present_eras(meta: Iterable[dict[str, Any]]) -> list[str]:
    """Eras that actually carry rows, in chronological order."""
    seen = {r["era"] for r in meta}
    return [era for era in ERAS if era in seen]


def verdict_mix_by_era(meta: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Share of each judge verdict within each era, over citations judged."""
    rows = list(meta)
    by_era: dict[str, Counter] = defaultdict(Counter)
    for record in rows:
        by_era[record["era"]][record["judge_status"]] += 1

    out: list[dict[str, Any]] = []
    for era in _present_eras(rows):
        counts = by_era[era]
        total = sum(counts.values())
        for status in sorted(counts):
            out.append({"era": era, "status": status, "n": counts[status],
                        "n_era": total, "share": counts[status] / total})
    return out


def grounding_by_era(meta: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Self-reported verification effort per era.

    This is effort across all tools, not a count of web searches: agents
    reported effort while also reporting that search was unavailable, having
    verified through Crossref and DOI resolution instead. A citation with no
    effort record is not a zero-effort citation, so it leaves both the
    numerator and the denominator.
    """
    rows = list(meta)
    out: list[dict[str, Any]] = []
    for era in _present_eras(rows):
        era_rows = [r for r in rows if r["era"] == era]
        counts = [r["judge_search_count"] for r in era_rows
                  if r["judge_search_count"] is not None]
        n_recorded = len(counts)
        zeros = sum(1 for c in counts if c == 0)
        out.append({
            "era": era,
            "n_judged": len(era_rows),
            "n_effort_recorded": n_recorded,
            "n_missing_effort": len(era_rows) - n_recorded,
            "n_zero_effort": zeros,
            "zero_effort_share": zeros / n_recorded if n_recorded else None,
            "mean_effort": sum(counts) / n_recorded if n_recorded else None,
        })
    return out


def composition_gaps(
    meta: Iterable[dict[str, Any]],
    factors: dict[str, Callable[[dict[str, Any]], str]],
) -> list[dict[str, Any]]:
    """Largest gap, in percentage points, between an era's mix and the overall mix.

    A small gap here is what makes the drift an instrument shift rather than a
    content shift: the eras were judging the same kind of material.
    """
    rows = list(meta)
    eras = _present_eras(rows)
    out: list[dict[str, Any]] = []
    for name, extract in factors.items():
        overall = Counter(extract(r) for r in rows)
        n_all = sum(overall.values())
        worst, worst_era, worst_level = 0.0, None, None
        for era in eras:
            era_rows = [r for r in rows if r["era"] == era]
            counts = Counter(extract(r) for r in era_rows)
            n_era = sum(counts.values())
            for level, overall_n in overall.items():
                gap = abs(100 * counts[level] / n_era - 100 * overall_n / n_all)
                if gap > worst:
                    worst, worst_era, worst_level = gap, era, level
        out.append({"factor": name, "max_gap_pp": worst,
                    "worst_era": worst_era, "worst_level": worst_level})
    return out


FABRICATED_STATUS = "FABRICATED"


def fab_share_by_effort_era(meta: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """FABRICATED share within each recorded effort level and era.

    This is the diagnostic that distinguishes the judge checking fewer things
    from the judge scoring the things it checked differently. Read it with the
    caveat that effort is endogenous: a citation that resolves on the first
    lookup never needs a third, so effort is chosen in response to what the
    judge is finding and stratifying on it does not cleanly hold difficulty
    fixed. It still bounds the coverage explanation, because a pure coverage
    failure would concentrate in the zero-effort rows.
    """
    rows = [r for r in meta if r["judge_search_count"] is not None]
    eras = _present_eras(rows)
    out: list[dict[str, Any]] = []
    for effort in sorted({r["judge_search_count"] for r in rows}):
        for era in eras:
            group = [r for r in rows
                     if r["judge_search_count"] == effort and r["era"] == era]
            if not group:
                continue
            n_fab = sum(1 for r in group if r["judge_status"] == FABRICATED_STATUS)
            out.append({"effort": effort, "era": era, "n": len(group),
                        "n_fabricated": n_fab, "fab_share": n_fab / len(group)})
    return out


def real_share_by_cell_era(
    meta: Iterable[dict[str, Any]],
    cell_of: Callable[[str], str],
) -> list[dict[str, Any]]:
    """REAL share within each question cell and era.

    The known-real cell is the diagnostic one. Its citations point at genuinely
    real work disproportionately often, so a fall in REAL there is the judge
    failing to confirm, not the models fabricating more.
    """
    rows = list(meta)
    by: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for record in rows:
        by[(cell_of(record["question_id"]), record["era"])][record["judge_status"]] += 1

    eras = _present_eras(rows)
    cells = [c for c in CELLS if any((c, e) in by for e in eras)]
    extra = sorted({c for c, _ in by} - set(CELLS))
    out: list[dict[str, Any]] = []
    for cell in cells + extra:
        for era in eras:
            counts = by.get((cell, era))
            if not counts:
                continue
            total = sum(counts.values())
            out.append({"cell": cell, "era": era, "n": total,
                        "n_real": counts[REAL_STATUS],
                        "real_share": counts[REAL_STATUS] / total,
                        "n_fabricated": counts[FABRICATED_STATUS],
                        "fab_share": counts[FABRICATED_STATUS] / total})
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_report(judgments: Path, questions: Path, models: Path) -> dict[str, Any]:
    meta = load_judgment_meta(judgments)
    cells = {r["id"]: r["cell"] for r in _read_jsonl(questions)}
    domains = {r["id"]: r["domain"] for r in _read_jsonl(questions)}
    registry = {r["model_id"]: r for r in _read_jsonl(models)}

    factors = {
        "cell": lambda r: cells[r["question_id"]],
        "domain": lambda r: domains[r["question_id"]],
        "condition": lambda r: r["condition"],
        "stage_ordinal": lambda r: str(registry[r["model_id"]]["stage_ordinal"]),
        "family": lambda r: registry[r["model_id"]]["family"],
    }
    return {
        "n_judged_citations": len(meta),
        "source_judgments": str(judgments).replace("\\", "/"),
        "verdict_mix": verdict_mix_by_era(meta),
        "grounding": grounding_by_era(meta),
        "composition_gaps": composition_gaps(meta, factors),
        "real_share_by_cell": real_share_by_cell_era(meta, lambda qid: cells[qid]),
        "fab_share_by_effort": fab_share_by_effort_era(meta),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--models", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    report = build_report(args.judgments, args.questions, args.models)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"judged citations: {report['n_judged_citations']}")
    for row in report["grounding"]:
        print(f"  {row['era']:<10} effort {row['mean_effort']:.4f}  "
              f"zero {100 * row['zero_effort_share']:.2f}%")
    worst = max(report["composition_gaps"], key=lambda r: r["max_gap_pp"])
    print(f"largest composition gap: {worst['max_gap_pp']:.2f} pts "
          f"({worst['factor']}, {worst['worst_era']}/{worst['worst_level']})")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
