"""Scratch: verifier-REAL citation population shape for the EXISTENT_WRONG_ID
sub-study (#3). Counts the verifier-REAL citations J2a TRUSTED (never judged),
broken out by (domain, cell) and by condition, so the 200-cit stratified
allocation can be chosen against the real distribution.
"""
from collections import Counter
from pathlib import Path

from src.judge_via_subagents import _build_qmap, build_judge_inputs

EXTR = Path("data/extractions/phase_b/extractions.jsonl")
VERIF = Path("data/verifications/phase_b/verifications.jsonl")
QMAP = Path("data/questions/frozen_900q_legal_amended.jsonl")

qmap = _build_qmap(QMAP)
inputs = list(build_judge_inputs(EXTR, VERIF, QMAP, fire_on_statuses=("REAL",)))

by_domain_cell = Counter()
by_domain = Counter()
by_condition = Counter()
for inp in inputs:
    row = qmap[inp.question_id]
    by_domain_cell[(row["domain"], row["cell"])] += 1
    by_domain[row["domain"]] += 1
    by_condition[inp.condition] += 1

total = len(inputs)
print(f"TOTAL verifier-REAL citations (J2a-trusted population): {total}\n")

print("by domain:")
for d, n in sorted(by_domain.items()):
    print(f"  {d:10s} {n:5d}  ({n/total:.1%})")

print("\nby (domain, cell) [9 strata]:")
for (d, c), n in sorted(by_domain_cell.items()):
    prop200 = 200 * n / total
    print(f"  {d:10s} {c:14s} {n:5d}  ({n/total:5.1%})  prop-alloc->{prop200:5.1f}")

print("\nby condition:")
for cond, n in sorted(by_condition.items()):
    print(f"  {cond:22s} {n:5d}  ({n/total:.1%})")
