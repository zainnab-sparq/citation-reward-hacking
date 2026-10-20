"""Scratch: decompose the legal x length_matched x unanswerable J2a FAB cell.

Theory-reviewer question (2026-06-11 panel): of the 271 generations labeled
fabricated in this cell, how many rest on judge-CONFIRMED FABRICATED verdicts
vs the UNV-fallback-to-verifier-FAB path (a potential measurement-floor
artifact)? Also reports the citation-level split and the same decomposition
for medical x length_matched x unanswerable (49.0% cell).
"""
import json
from collections import Counter
from pathlib import Path

from src.analysis.aggregate import (
    _read_jsonl,
    apply_j2a,
    build_judgment_index,
    flatten_verifications,
    to_questions_frame,
)

QUESTIONS = Path("data/questions/frozen_900q_legal_amended.jsonl")
VERIF = Path("data/verifications/phase_b/verifications.jsonl")
JUDG = Path("data/judgments/phase_b/claude-subagent-sonnet.jsonl")

CELLS = [("legal", "length_matched", "unanswerable"),
         ("medical", "length_matched", "unanswerable")]

questions = to_questions_frame(QUESTIONS)
cell_domain = {q["id"]: (q["cell"], q["domain"]) for q in questions}
judgment_index = build_judgment_index(_read_jsonl(JUDG))
flat = list(flatten_verifications(_read_jsonl(VERIF)))
rows = list(apply_j2a(flat, judgment_index))

for domain, condition, cell in CELLS:
    by_gen: dict[tuple, list[dict]] = {}
    for r in rows:
        cd = cell_domain.get(r["question_id"])
        if cd != (cell, domain) or r["condition"] != condition:
            continue
        key = (r["question_id"], r["model_id"], r["trial_id"])
        by_gen.setdefault(key, []).append(r)

    gen_kinds = Counter()
    cit_kinds = Counter()
    for key, gen_rows in by_gen.items():
        bad = [r for r in gen_rows if r["status"] == "FABRICATED"]
        if not bad:
            continue
        kinds = set()
        for r in bad:
            jkey = (r["question_id"], r["model_id"], r["condition"],
                    r["trial_id"], r["citation_index"])
            verdict = judgment_index.get(jkey)
            if verdict == "FABRICATED":
                kind = f"judge_FAB_on_verifier_{r['verifier_status'][:3]}"
            else:
                kind = f"fallback_verifier_FAB_judge_{verdict}"
            kinds.add(kind)
            cit_kinds[kind] += 1
        if all(k.startswith("judge_FAB") for k in kinds):
            gen_kinds["all_bad_citations_judge_confirmed"] += 1
        elif all(k.startswith("fallback") for k in kinds):
            gen_kinds["all_bad_citations_fallback"] += 1
        else:
            gen_kinds["mixed"] += 1

    print(f"\n=== {domain} x {condition} x {cell} ===")
    print("fabricated generations:", sum(gen_kinds.values()))
    print("generation-level:", json.dumps(gen_kinds, indent=1))
    print("citation-level (bad citations only):", json.dumps(cit_kinds, indent=1))
