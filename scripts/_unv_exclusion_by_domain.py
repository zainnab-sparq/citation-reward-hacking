"""Generation-level UNVERIFIABLE exclusion rate by domain, for the limitations section.

`unverifiable_selection_check` reports the domain breakdown only inside the
`obscure_real` trigger cell, because that is the cell the registered §6
escalation rule is written about. The limitations section needs the wider
number: across every cell, how often does the NA-exclusion drop a whole
generation, and how unevenly does that fall across academic, legal and medical?

That matters because the registered E1 family tests two domain three-way terms
(`stage_ordinal:cellobscure_real:domainmedical` and `:domainlegal`). If the
exclusion rate differs by domain, the domain terms are estimated on differently
complete data, so the missingness is not ignorable by construction and has to be
disclosed as such.

Same exclusion rule as the registered check, imported rather than reimplemented.
Diagnostic only: it reports WHO is missing and never a fabrication rate.
"""
import json
from pathlib import Path

from src.analysis.unverifiable_selection_check import (
    build_judgment_index,
    cell_of_question_id,
    domain_of_question_id,
    generation_final_statuses,
    is_excluded,
    read_jsonl,
)

VERIFICATIONS = Path("data/verifications/phase_c/verifications.jsonl")
JUDGMENTS = Path("data/judgments/phase_c/claude-subagent-sonnet.jsonl")
OUTPUT = Path("data/analysis/phase_c/unv_exclusion_by_domain.json")

DOMAINS = ("academic", "legal", "medical")
CELLS = ("known_real", "obscure_real", "unanswerable")


def _rate(excluded, n):
    return round(100 * excluded / n, 4) if n else None


index = build_judgment_index(read_jsonl(JUDGMENTS))

totals = {domain: [0, 0] for domain in DOMAINS}
by_cell = {domain: {cell: [0, 0] for cell in CELLS} for domain in DOMAINS}
n_scored = 0
n_skipped = 0

for record in read_jsonl(VERIFICATIONS):
    statuses = record.get("citation_statuses") or []
    if not statuses:
        continue
    final, complete = generation_final_statuses(
        record["question_id"], record["model_id"], record["condition"],
        record["trial_id"], statuses, index,
    )
    if not complete:
        n_skipped += 1
        continue
    n_scored += 1
    excluded = int(is_excluded(final))
    domain = domain_of_question_id(record["question_id"])
    cell = cell_of_question_id(record["question_id"])
    totals[domain][0] += 1
    totals[domain][1] += excluded
    by_cell[domain][cell][0] += 1
    by_cell[domain][cell][1] += excluded

print(f"scored generations: {n_scored}   skipped (incomplete): {n_skipped}")
print()
print(f"{'domain':<10} {'n':>7} {'excluded':>9} {'rate':>8}")
for domain in DOMAINS:
    n, excluded = totals[domain]
    print(f"{domain:<10} {n:>7} {excluded:>9} {_rate(excluded, n):>7}%")

print()
print(f"{'domain':<10} {'cell':<14} {'n':>7} {'excluded':>9} {'rate':>8}")
for domain in DOMAINS:
    for cell in CELLS:
        n, excluded = by_cell[domain][cell]
        print(f"{domain:<10} {cell:<14} {n:>7} {excluded:>9} {_rate(excluded, n):>7}%")

payload = {
    "n_scored": n_scored,
    "n_skipped_incomplete": n_skipped,
    "by_domain": {
        domain: {"n": totals[domain][0], "excluded": totals[domain][1],
                 "rate": _rate(totals[domain][1], totals[domain][0])}
        for domain in DOMAINS
    },
    "by_domain_cell": {
        domain: {
            cell: {"n": by_cell[domain][cell][0],
                   "excluded": by_cell[domain][cell][1],
                   "rate": _rate(by_cell[domain][cell][1], by_cell[domain][cell][0])}
            for cell in CELLS
        }
        for domain in DOMAINS
    },
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
print(f"\nwritten: {OUTPUT}")
