"""Generation-level UNVERIFIABLE exclusion rate by judging regime.

The drift disclosure adds regime as a third stratification of the registered
exclusion table, because the citation-level UNVERIFIABLE rate moved a long way
across regimes and regime did not exist at registration. This computes the
generation-level version of that, which is much smaller than the citation-level
rate: a generation is excluded only when EVERY one of its citations is
UNVERIFIABLE.

Regime comes from `judge_era.py`, which assigns it per generation from the
position of that generation's judgments in the append-only judgment file. A
generation whose citations straddle a boundary is MIXED and is reported
separately rather than forced into a regime.

Diagnostic only: it reports who is missing, never a fabrication rate, and it
carries no stage or condition term.
"""
import collections
import json
from pathlib import Path

from src.analysis.unverifiable_selection_check import (
    build_judgment_index,
    generation_final_statuses,
    is_excluded,
    read_jsonl,
)

VERIFICATIONS = Path("data/verifications/phase_c/verifications.jsonl")
JUDGMENTS = Path("data/judgments/phase_c/claude-subagent-sonnet.jsonl")
ERA = Path("data/analysis_inputs/phase_c_final/judgment_era.jsonl")
OUTPUT = Path("data/analysis/phase_c/unv_exclusion_by_era.json")

GEN_KEYS = ("question_id", "model_id", "condition", "trial_id")
# A generation whose citations the structured verifier all confirmed never
# reaches the judge, so it has no position in the judgment file and no regime.
# It also cannot be UNVERIFIABLE-excluded. Reported rather than dropped.
UNJUDGED = "NO_JUDGED_CITATION"
ERA_ORDER = ("EARLY", "DEGRADED", "RECOVERED", "MIXED", UNJUDGED)

era_of = {
    tuple(row[key] for key in GEN_KEYS): row["era"]
    for row in read_jsonl(ERA)
}
index = build_judgment_index(read_jsonl(JUDGMENTS))

counts = collections.defaultdict(lambda: [0, 0])
n_scored = 0
for record in read_jsonl(VERIFICATIONS):
    statuses = record.get("citation_statuses") or []
    if not statuses:
        continue
    key = tuple(record[k] for k in GEN_KEYS)
    final, complete = generation_final_statuses(*key, statuses, index)
    if not complete:
        continue
    n_scored += 1
    bucket = counts[era_of.get(key, UNJUDGED)]
    bucket[0] += 1
    bucket[1] += int(is_excluded(final))

print(f"scored generations: {n_scored}")
print()
print(f"{'regime':<10} {'n':>7} {'excluded':>9} {'rate':>8}")
payload = {"n_scored": n_scored, "by_era": {}}
for era in ERA_ORDER:
    n, excluded = counts[era]
    rate = round(100 * excluded / n, 4) if n else None
    payload["by_era"][era] = {"n": n, "excluded": excluded, "rate": rate}
    shown = f"{rate:.2f}%" if rate is not None else "n/a"
    print(f"{era:<10} {n:>7} {excluded:>9} {shown:>8}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"\nwritten: {OUTPUT}")
