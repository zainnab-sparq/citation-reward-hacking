"""Pre-registered selection check on cited-only-UNVERIFIABLE exclusions (§6 sensitivity #1).

The registered primary excludes a generation that cited ONLY UNVERIFIABLE sources,
because it carries neither a good nor a bad citation and so has no fabrication
outcome. That exclusion conditions on "the model cited something verifiable",
which is post-treatment if UNVERIFIABLE rates differ by stage or condition.

Pre-registration §6 sensitivity #1 therefore commits, BEFORE the primary result is
interpreted, to reporting the exclusion rate stratified by `cell x stage_ordinal`
(E1) and `condition x is_post_trained` (E2), with an escalation rule:

    if that rate is monotone in stage/condition WITHIN the obscure-but-real cell
    (a sign the exclusion is informative rather than ignorable), the
    UNVERIFIABLE-as-fabricated sensitivity is elevated from a footnote to a
    CO-PRIMARY result reported alongside the exclusion-based primary.

This module computes exactly that table and evaluates the trigger. It is a
MISSINGNESS DIAGNOSTIC: it reports who is excluded, never a fabrication rate by
condition, so it reveals no registered estimand and is safe to run before the
one-shot confirmatory fit. That is why it carries no `--robustness-panel`-style
gate, unlike `phase_c_drift_interaction.R`'s panel.

Exclusion is evaluated on the J2a-combined label, which is the operative coding of
the primary. The load-bearing detail is that a judge verdict of UNVERIFIABLE does
NOT rescue a citation: per `aggregate.combine_j2a`, the verifier label stands, so
the generation remains excluded. Only judge-REAL or judge-FABRICATED resolves it.

Monotonicity is evaluated on the stage axis only. `stage_ordinal` is ordered
(base < SFT < DPO < RLVR/Instruct); `condition` is a nominal factor with no
ordering, so "monotone in condition" is undefined and the E2 table is reported
descriptively for the reader rather than run through the trigger.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

CELLS = ("obscure_real", "known_real", "unanswerable")
STAGES = (0, 1, 2, 3)
CONDITIONS = ("neutral", "citations_expected", "length_matched", "abstention_permitted")
TRIGGER_CELL = "obscure_real"
UNVERIFIABLE = "UNVERIFIABLE"


def cell_of_question_id(question_id: str) -> str:
    """`<domain>_<cell>_<index>`. Checked longest-first so `obscure_real` is never
    mistaken for `known_real` (both end in `_real`)."""
    for cell in CELLS:
        if f"_{cell}_" in question_id:
            return cell
    raise ValueError(f"no recognised cell in question_id {question_id!r}")


def domain_of_question_id(question_id: str) -> str:
    return question_id.split("_", 1)[0]


def parse_custom_id(custom_id: str) -> tuple[str, str, str, int, int]:
    """`<question_id>__<model_id>__<condition>__t<trial>__c<citation_index>`.

    Split on '__' and never on '/': open-weights model ids embed a slash.
    """
    question_id, model_id, condition, trial, citation = custom_id.split("__")
    return question_id, model_id, condition, int(trial[1:]), int(citation[1:])


def build_judgment_index(
    judgments: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str, int, int], str]:
    """Index judge verdicts by citation key, skipping errored rows.

    An errored row is dropped rather than stored as None, so the generation reads
    as NOT-yet-judged and is skipped by the caller. Treating it as a verdict would
    silently code an API failure as a verifier fallback.
    """
    index: dict[tuple[str, str, str, int, int], str] = {}
    for record in judgments:
        if record.get("error") is not None:
            continue
        status = record.get("judge_status")
        if status is None:
            continue
        index[parse_custom_id(record["custom_id"])] = status
    return index


def generation_final_statuses(
    question_id: str,
    model_id: str,
    condition: str,
    trial_id: int,
    verifier_statuses: Sequence[str],
    judgment_index: dict[tuple[str, str, str, int, int], str],
) -> tuple[list[str] | None, bool]:
    """Apply the J2a combine rule to one generation's citations.

    Returns `(final_statuses, complete)`. `complete` is False when any citation
    that should have fired the judge has no verdict yet, in which case the
    generation must be SKIPPED rather than scored: judging proceeds in file order,
    so scoring partially-judged generations would bias the rate by judging order.
    """
    final: list[str] = []
    for index, status in enumerate(verifier_statuses):
        if status.startswith("REAL"):
            final.append("REAL")
            continue
        verdict = judgment_index.get((question_id, model_id, condition, trial_id, index))
        if verdict is None:
            return None, False
        if verdict == "REAL":
            final.append("REAL")
        elif verdict == "FABRICATED":
            final.append("FABRICATED")
        else:
            final.append(status)  # judge UNVERIFIABLE -> verifier label stands
    return final, True


def is_excluded(final_statuses: Sequence[str]) -> bool:
    """True when the generation cited only UNVERIFIABLE sources.

    An empty citation list is NOT an exclusion: citing nothing is a separate
    outcome (abstention), handled elsewhere in the registered coding.
    """
    return bool(final_statuses) and all(s == UNVERIFIABLE for s in final_statuses)


def monotone_direction(values: Sequence[float | None]) -> str:
    """`UP` / `DOWN` / `FLAT` / `NO` over the non-missing values, in order.

    FLAT is reported separately from UP/DOWN so a constant sequence is never read
    as a trend; the registered trigger is about a rate that MOVES with stage.
    """
    seq = [v for v in values if v is not None]
    if len(seq) < 2:
        return "NO"
    if all(a == b for a, b in zip(seq, seq[1:])):
        return "FLAT"
    if all(a <= b for a, b in zip(seq, seq[1:])):
        return "UP"
    if all(a >= b for a, b in zip(seq, seq[1:])):
        return "DOWN"
    return "NO"


def _rate(excluded: int, n: int) -> float | None:
    return round(excluded / n * 100, 4) if n else None


def _cellify(counts: dict[Any, int], totals: dict[Any, int], key: Any) -> dict[str, Any]:
    n = totals.get(key, 0)
    excluded = counts.get(key, 0)
    return {"n": n, "excluded": excluded, "rate": _rate(excluded, n)}


def selection_check(
    verifications: Iterable[dict[str, Any]],
    judgments: Iterable[dict[str, Any]],
    registry: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compute the §6 sensitivity #1 exclusion tables and evaluate the trigger."""
    stage_of: dict[str, int] = {}
    post_of: dict[str, int] = {}
    for row in registry:
        stage_of[row["model_id"]] = row["stage_ordinal"]
        post_of[row["model_id"]] = row["is_post_trained"]

    index = build_judgment_index(judgments)

    e1_n: dict[tuple[str, int], int] = {}
    e1_x: dict[tuple[str, int], int] = {}
    e2_n: dict[tuple[str, int], int] = {}
    e2_x: dict[tuple[str, int], int] = {}
    dom_n: dict[tuple[str, int], int] = {}
    dom_x: dict[tuple[str, int], int] = {}
    n_scored = 0
    n_skipped = 0
    n_no_citations = 0

    for record in verifications:
        statuses = record.get("citation_statuses") or []
        if not statuses:
            n_no_citations += 1
            continue
        question_id = record["question_id"]
        model_id = record["model_id"]
        condition = record["condition"]
        trial_id = record["trial_id"]
        final, complete = generation_final_statuses(
            question_id, model_id, condition, trial_id, statuses, index
        )
        if not complete:
            n_skipped += 1
            continue
        n_scored += 1
        excluded = int(is_excluded(final))
        cell = cell_of_question_id(question_id)
        domain = domain_of_question_id(question_id)
        stage = stage_of[model_id]
        post = post_of[model_id]

        e1_n[(cell, stage)] = e1_n.get((cell, stage), 0) + 1
        e1_x[(cell, stage)] = e1_x.get((cell, stage), 0) + excluded
        e2_n[(condition, post)] = e2_n.get((condition, post), 0) + 1
        e2_x[(condition, post)] = e2_x.get((condition, post), 0) + excluded
        if cell == TRIGGER_CELL:
            dom_n[(domain, stage)] = dom_n.get((domain, stage), 0) + 1
            dom_x[(domain, stage)] = dom_x.get((domain, stage), 0) + excluded

    e1 = {
        cell: {str(stage): _cellify(e1_x, e1_n, (cell, stage)) for stage in STAGES}
        for cell in CELLS
    }
    e2 = {
        condition: {str(post): _cellify(e2_x, e2_n, (condition, post)) for post in (0, 1)}
        for condition in CONDITIONS
    }

    pooled = [e1[TRIGGER_CELL][str(stage)]["rate"] for stage in STAGES]
    pooled_direction = monotone_direction(pooled)

    per_domain: dict[str, Any] = {}
    for domain in sorted({d for d, _ in dom_n}):
        rates = [_rate(dom_x.get((domain, s), 0), dom_n.get((domain, s), 0)) for s in STAGES]
        per_domain[domain] = {
            "rates": rates,
            "n": [dom_n.get((domain, s), 0) for s in STAGES],
            "direction": monotone_direction(rates),
        }

    return {
        "n_scored": n_scored,
        "n_skipped_incomplete": n_skipped,
        "n_no_citations": n_no_citations,
        "e1_cell_by_stage": e1,
        "e2_condition_by_post_trained": e2,
        "escalation": {
            "trigger_cell": TRIGGER_CELL,
            "pooled_rates_by_stage": pooled,
            "pooled_direction": pooled_direction,
            "registered_trigger_fires": pooled_direction in ("UP", "DOWN"),
            "per_domain": per_domain,
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pre-registered UNVERIFIABLE-exclusion selection check (§6 sensitivity #1)."
    )
    parser.add_argument("--verifications", required=True, type=Path)
    parser.add_argument("--judgments", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    result = selection_check(
        read_jsonl(args.verifications),
        read_jsonl(args.judgments),
        read_jsonl(args.registry),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    escalation = result["escalation"]
    print(f"scored generations:      {result['n_scored']}")
    print(f"skipped (not yet judged): {result['n_skipped_incomplete']}")
    print(f"pooled {TRIGGER_CELL} rates by stage: {escalation['pooled_rates_by_stage']}")
    print(f"pooled direction:        {escalation['pooled_direction']}")
    print(f"REGISTERED TRIGGER FIRES: {escalation['registered_trigger_fires']}")
    for domain, info in escalation["per_domain"].items():
        print(f"  {domain:<9} {info['direction']:<5} rates={info['rates']} n={info['n']}")
    print(f"written: {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
