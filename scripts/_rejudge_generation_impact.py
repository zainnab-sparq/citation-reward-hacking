"""How many generation-level outcomes does the verifier-REAL correction move?

Fills the counts §11 sub-amendment #12 promises. It compares the two built input
sets at identical judgments, one with the correction applied and one without,
and reports how many generations change their registered `fabricated` outcome.

Deliberately reports TOTALS ONLY. Breaking the flip count down by condition
would be the amendment #10 delta, and by stage would be the E1 cascade. Both are
one-shot registered estimands and neither is computed here.

Coding matches the registered outcome exactly: a generation is `fabricated` when
any of its citations is FABRICATED, and is NA-excluded when it cites only
UNVERIFIABLE sources. Generations with no citations are neither.
"""
import collections
import json
from pathlib import Path

REGISTERED = Path("data/analysis_inputs/phase_c_registered/verifications.jsonl")
CORRECTED = Path("data/analysis_inputs/phase_c_final/verifications.jsonl")

GEN_KEYS = ("question_id", "model_id", "condition", "trial_id")


def _outcomes(path):
    """generation key -> 'fabricated' | 'clean' | 'excluded'."""
    statuses = collections.defaultdict(list)
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        statuses[tuple(row[k] for k in GEN_KEYS)].append(row["status"])
    out = {}
    for key, values in statuses.items():
        if any(s == "FABRICATED" for s in values):
            out[key] = "fabricated"
        elif all(s == "UNVERIFIABLE" for s in values):
            out[key] = "excluded"
        else:
            out[key] = "clean"
    return out


registered = _outcomes(REGISTERED)
corrected = _outcomes(CORRECTED)
assert registered.keys() == corrected.keys(), "generation sets differ"

transitions = collections.Counter(
    (registered[key], corrected[key]) for key in registered
)
n = len(registered)
print(f"citing generations: {n}")
print()
print(f"{'registered':<12} -> {'corrected':<12} {'n':>7}")
for (before, after), count in sorted(transitions.items(), key=lambda kv: -kv[1]):
    marker = "   <- moved" if before != after else ""
    print(f"{before:<12} -> {after:<12} {count:>7}{marker}")

moved = sum(count for (before, after), count in transitions.items() if before != after)
fab_before = sum(count for (before, _), count in transitions.items()
                 if before == "fabricated")
fab_after = sum(count for (_, after), count in transitions.items()
                if after == "fabricated")
print()
print(f"generations moved:        {moved} ({100 * moved / n:.2f}% of citing generations)")
print(f"fabricated, registered:   {fab_before} ({100 * fab_before / n:.2f}%)")
print(f"fabricated, corrected:    {fab_after} ({100 * fab_after / n:.2f}%)")
print(f"absolute shift:           {100 * (fab_after - fab_before) / n:+.2f}pp")
