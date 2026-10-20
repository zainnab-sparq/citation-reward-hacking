"""Paired re-judge sub-study: sample already-judged citations and re-judge them.

The J2a judge's behaviour changed twice during collection. Observational era
comparisons cannot separate "the instrument changed" from "those items differ",
because each era contains different citations. This sub-study removes that
ambiguity by holding the item fixed: the same citations are judged again under
the current instrument, and BOTH verdicts are kept, giving a paired transition
matrix per condition.

Every original prompt file is still on disk, so the re-judge reads the
BYTE-IDENTICAL prompt the original judgment read. Prompt-construction variance is
eliminated by construction rather than assumed away.

Design (docs/drift_interaction_prespec.md §9):
  * sample from EARLY and DEGRADED. RECOVERED rows were produced by the current
    instrument, so re-judging them measures noise, not drift.
  * stratify by condition, since differentiality BY CONDITION is the question.
  * write to a SEPARATE output file. The main checkpoint is append-only with no
    dedup-on-write, so writing re-judgments into it would corrupt the dataset and
    destroy the pairing.

Usage:
  python -m src.analysis.substudy_rejudge \
      --judgments data/judgments/phase_c/claude-subagent-sonnet.jsonl \
      --cache ~/.cache/citation-judge-prompts-phase-c \
      --output data/judgments/phase_c/rejudge_substudy.jsonl \
      --n-per-era 250

Then run the emitted batch script via the Workflow tool and persist with
`src.run_judge_workflow persist --output <the sub-study file>`.
"""
from __future__ import annotations

import argparse
import json
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.analysis.judge_era import load_judgment_meta
from src.judge_via_workflow import _prompt_filename, successful_custom_ids

# The batch-script template is imported rather than copied so the re-judge
# dispatches through exactly the same agent configuration (sonnet, WebSearch +
# WebFetch, same schema) as the original judgment. A divergence here would make
# the comparison measure the harness instead of the instrument.
from src.run_judge_workflow import _WORKFLOW_TEMPLATE

SUBSTUDY_SEED = 20260811
REJUDGE_ERAS = ("EARLY", "DEGRADED")
DEFAULT_N_PER_ERA = 250


def allocate_by_condition(counts: dict[str, int], total: int) -> dict[str, int]:
    """Largest-remainder proportional allocation, capped at what is available."""
    available = sum(counts.values())
    target = min(total, available)
    exact = {k: (v / available) * target if available else 0.0 for k, v in counts.items()}
    alloc = {k: min(int(v), counts[k]) for k, v in exact.items()}
    remaining = target - sum(alloc.values())
    if remaining > 0:
        order = sorted(counts, key=lambda k: (-(exact[k] - int(exact[k])), k))
        for key in order:
            if remaining == 0:
                break
            if alloc[key] < counts[key]:
                alloc[key] += 1
                remaining -= 1
    return alloc


def allocate_equally(counts: dict[str, int], total: int) -> dict[str, int]:
    """Balanced allocation across strata, spilling over from exhausted ones.

    Balanced is the powerful design here because the sub-study exists to compare
    flip rates ACROSS conditions, and the variance of that comparison is driven
    by the smallest cell. Proportional allocation would put abstention_permitted
    at about 29 per era against citations_expected's 90; balancing lifts the
    floor to about 62. Corpus-level rates are recovered with the design weights
    attached to each sampled item.
    """
    alloc = {key: 0 for key in counts}
    remaining = min(total, sum(counts.values()))
    while remaining > 0:
        open_strata = [k for k in sorted(counts) if alloc[k] < counts[k]]
        if not open_strata:
            break
        share = max(1, remaining // len(open_strata))
        for key in open_strata:
            if remaining == 0:
                break
            take = min(share, counts[key] - alloc[key], remaining)
            alloc[key] += take
            remaining -= take
    return alloc


_ALLOCATORS = {"equal": allocate_equally, "proportional": allocate_by_condition}


def sample_rejudge_items(
    meta: Iterable[dict[str, Any]],
    n_per_era: int = DEFAULT_N_PER_ERA,
    seed: int = SUBSTUDY_SEED,
    exclude: Iterable[str] | None = None,
    allocation: str = "equal",
) -> list[dict[str, Any]]:
    """Stratified sample of judged citations to re-judge, one list across eras.

    Each item carries a design weight (stratum population / stratum sample) so a
    corpus-level rate can be recovered from a balanced sample, the same design
    used by the Phase B EXISTENT_WRONG_ID sub-study.
    """
    excluded = set(exclude or ())
    pool: dict[str, dict[str, list[dict[str, Any]]]] = {era: {} for era in REJUDGE_ERAS}
    for record in meta:
        if record["era"] not in pool or record["custom_id"] in excluded:
            continue
        pool[record["era"]].setdefault(record["condition"], []).append(record)

    allocate = _ALLOCATORS[allocation]
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    for era in REJUDGE_ERAS:
        strata = pool[era]
        alloc = allocate({k: len(v) for k, v in strata.items()}, n_per_era)
        for condition in sorted(strata):
            n_drawn = alloc[condition]
            if not n_drawn:
                continue
            chosen = rng.sample(strata[condition], n_drawn)
            weight = len(strata[condition]) / n_drawn
            for record in sorted(chosen, key=lambda r: r["row_index"]):
                items.append(
                    {
                        "custom_id": record["custom_id"],
                        "era": era,
                        "condition": condition,
                        "row_index": record["row_index"],
                        "original_status": record["judge_status"],
                        "original_search_count": record["judge_search_count"],
                        "stratum_population": len(strata[condition]),
                        "design_weight": weight,
                    }
                )
    return items


def prompt_path_for(custom_id: str, prompts_dir: Path) -> str:
    return str(Path(prompts_dir) / _prompt_filename(custom_id))


def build_batch_script(items: Sequence[dict[str, Any]], name: str) -> str:
    """Render the Workflow batch script. Only custom_id and prompt_path travel
    into the script; the original verdict stays in the manifest so it cannot
    leak into the judge's context and bias the re-judgment."""
    payload = [
        {"custom_id": item["custom_id"], "prompt_path": item["prompt_path"]}
        for item in items
    ]
    return _WORKFLOW_TEMPLATE.format(
        name=name,
        name_json=json.dumps(name),
        items_json=json.dumps(payload),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the paired re-judge sub-study batch.")
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Sub-study judgments JSONL. MUST NOT be the main checkpoint.")
    parser.add_argument("--n-per-era", type=int, default=DEFAULT_N_PER_ERA)
    parser.add_argument("--seed", type=int, default=SUBSTUDY_SEED)
    parser.add_argument("--allocation", choices=sorted(_ALLOCATORS), default="equal")
    args = parser.parse_args(argv)

    if args.output.resolve() == args.judgments.resolve():
        parser.error("--output must differ from --judgments: the main checkpoint is "
                     "append-only with no dedup-on-write, and overwriting it would "
                     "destroy the pairing.")

    prompts_dir = args.cache / "prompts"
    meta = load_judgment_meta(args.judgments)
    already = successful_custom_ids(args.output) if args.output.exists() else set()
    items = sample_rejudge_items(meta, args.n_per_era, args.seed, exclude=already,
                                 allocation=args.allocation)

    missing = []
    for item in items:
        path = prompt_path_for(item["custom_id"], prompts_dir)
        if not Path(path).exists():
            missing.append(item["custom_id"])
        item["prompt_path"] = path
    if missing:
        print(f"ERROR: {len(missing)} prompt files missing, e.g. {missing[:3]}")
        return 2

    batches_dir = args.cache / "substudy_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    script_path = batches_dir / "rejudge_000.js"
    script_path.write_text(build_batch_script(items, "rejudge-000"),
                           encoding="utf-8", newline="\n")

    manifest_path = batches_dir / "rejudge_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    by_cell: dict[tuple[str, str], int] = {}
    for item in items:
        key = (item["era"], item["condition"])
        by_cell[key] = by_cell.get(key, 0) + 1
    print(f"sampled {len(items)} items ({len(already)} already re-judged, excluded)")
    for (era, condition), n in sorted(by_cell.items()):
        print(f"  {era:<10} {condition:<22} {n}")
    print(f"\nbatch script : {script_path}")
    print(f"manifest     : {manifest_path}")
    print(f"persist to   : {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
