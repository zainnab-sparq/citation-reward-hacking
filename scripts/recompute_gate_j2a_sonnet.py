"""Recompute §8 A.5 gate under J2a using the FORCED-SONNET subagent judge stack.

This is the pre-Step-1 re-validation gate from amendment #6 (d): the existing
96% subagent gate was on Opus 4.7 (parent-session inheritance at pilot time).
Amendment #6 (d) requires every Phase B Agent dispatch to pin model="sonnet"
to match the registered Sonnet 4.6 judge. This script re-runs the J2a
restricted-firing combine rule against the new sonnet judgments to confirm
the >=90% gate still clears on the registered model.

Combine rule under J2a (unchanged from recompute_gate_j2a_restricted_firing.py):
  if verifier == REAL: take verifier (no judge call)
  elif judge in {REAL, FABRICATED} (and not judge error): take judge
  else: fall back to verifier
"""
import io
import json
from pathlib import Path

import pandas as pd

BLIND = Path("data/handcheck/step0_blind_handcheck.csv")
REVEAL = Path("data/handcheck/step0_verifier_labels_reveal.csv")
JUDGMENTS = Path("data/judgments/pilot_b/claude-subagent-sonnet.jsonl")
OUT = Path("data/handcheck/step0_gate_summary_j2a_sonnet.json")
AUDIT_LOG = Path("data/handcheck/step0_gate_result.txt")

_VALID = {"REAL", "FABRICATED", "UNVERIFIABLE"}


def canon_human(s):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    return {"FAB": "FABRICATED", "UNV": "UNVERIFIABLE"}.get(s, s) if s in (_VALID | {"FAB", "UNV"}) else None


def canon_verifier(s):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    if s == "REAL_CONFIRMED":
        return "REAL"
    return s if s in _VALID else s


def canon_judge(s):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    return s if s in _VALID else None


def collapse_binary(label):
    if label == "FABRICATED":
        return "FAB"
    if label in {"REAL", "UNVERIFIABLE"}:
        return "NOT_FAB"
    return None


def main():
    df = pd.read_csv(BLIND, encoding="utf-8")
    reveal = pd.read_csv(REVEAL, encoding="utf-8")
    merged = df.merge(reveal, on="row_id", how="left", validate="one_to_one")
    merged["_human"] = merged["your_status"].map(canon_human)
    merged["_verifier"] = merged["verifier_status"].map(canon_verifier)

    judgments = [json.loads(line) for line in JUDGMENTS.read_text(encoding="utf-8").splitlines() if line.strip()]
    judge_by_row = {int(j["row_id"]): j for j in judgments}

    js, jerr = [], []
    for _, row in merged.iterrows():
        j = judge_by_row.get(int(row["row_id"]))
        if j is None:
            js.append(None); jerr.append("missing_judgment"); continue
        js.append(canon_judge(j.get("judge_status")))
        jerr.append(j.get("error"))
    merged["_judge"] = js
    merged["_judge_error"] = jerr

    def combine_j2a(row):
        v = row["_verifier"]; j = row["_judge"]; je = row["_judge_error"]
        if v == "REAL":
            return v
        if j is None or j == "UNVERIFIABLE" or je:
            return v
        return j

    merged["_hybrid_j2a"] = merged.apply(combine_j2a, axis=1)

    n_judged_under_j1 = merged["_judge"].notna().sum()
    n_judged_under_j2a = (merged["_verifier"] != "REAL").sum()
    n_skipped_judges = (merged["_verifier"] == "REAL").sum()

    pairs = [(h, u) for h, u in zip(merged["_hybrid_j2a"], merged["_human"]) if h and u]
    n3 = len(pairs); a3 = sum(1 for h, u in pairs if h == u)
    bin_pairs = [(collapse_binary(h), collapse_binary(u)) for h, u in pairs]
    bn = sum(1 for x, y in bin_pairs if x and y); ba = sum(1 for x, y in bin_pairs if x == y)

    summary = {
        "labeler": "j2a_restricted_firing_subagent_judge_FORCED_SONNET",
        "description": "Subagent judge fires only on verifier-FAB/UNV; every Agent dispatch pinned model=\"sonnet\" per amendment #6 (d)",
        "judgments_file": str(JUDGMENTS),
        "n_judged_under_j1_baseline": int(n_judged_under_j1),
        "n_judged_under_j2a": int(n_judged_under_j2a),
        "n_judge_calls_skipped_by_j2a": int(n_skipped_judges),
        "pct_judge_calls_saved": round(100 * n_skipped_judges / n_judged_under_j1, 1) if n_judged_under_j1 else 0,
        "hybrid_3way": {"agree": a3, "n": n3, "pct": round(100 * a3 / n3, 1) if n3 else 0},
        "hybrid_binary": {"agree": ba, "n": bn, "pct": round(100 * ba / bn, 1) if bn else 0},
        "gate_threshold_3way_pct": 90.0,
        "gate_passed": (100 * a3 / n3 if n3 else 0) >= 90.0,
    }
    OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    disagree = merged[
        merged["_human"].notna() & merged["_hybrid_j2a"].notna()
        & (merged["_hybrid_j2a"] != merged["_human"])
    ]
    buf = io.StringIO()
    buf.write("\n" + "#" * 70 + "\n")
    buf.write("###  GATE RECOMPUTE UNDER J2a + FORCED-SONNET SUBAGENT JUDGE\n")
    buf.write("###  (amendment #6 (d): every Agent dispatch pinned model=\"sonnet\";\n")
    buf.write("###   pre-Step-1 re-validation of the 90% gate)\n")
    buf.write("#" * 70 + "\n\n")
    buf.write(f"Judgments loaded: {summary['n_judged_under_j1_baseline']}\n")
    buf.write(f"Judge calls under J2a: {summary['n_judged_under_j2a']} "
              f"(saved {summary['n_judge_calls_skipped_by_j2a']} = "
              f"{summary['pct_judge_calls_saved']}%)\n\n")
    buf.write("########################################################\n")
    buf.write("###  §8 A.5 GATE (J2a + SONNET): HYBRID vs HUMAN     ###\n")
    buf.write("########################################################\n")
    buf.write(f"=== HYBRID(j2a, forced sonnet) vs HUMAN (n={summary['hybrid_3way']['n']}) ===\n")
    buf.write(f"3-way agreement: {summary['hybrid_3way']['agree']}/{summary['hybrid_3way']['n']} = "
              f"{summary['hybrid_3way']['pct']}%\n")
    buf.write(f"Binary FAB-vs-NOT agreement: {summary['hybrid_binary']['agree']}/"
              f"{summary['hybrid_binary']['n']} = {summary['hybrid_binary']['pct']}%\n")
    buf.write(f"Gate threshold (3-way): {summary['gate_threshold_3way_pct']}%\n")
    buf.write(f"Gate passed: {summary['gate_passed']}\n\n")
    buf.write("########################################################\n")
    buf.write(f"###  J2a-sonnet-vs-human disagreements (n={len(disagree)})\n")
    buf.write("########################################################\n")
    for _, row in disagree.iterrows():
        title = row["title"] if isinstance(row["title"], str) else (row["case_name"] if isinstance(row["case_name"], str) else "")
        buf.write(
            f"  pres#{row['presentation_order']:>2} row{row['row_id']:>3} "
            f"{row['domain']:<8} {row['cell']:<14} "
            f"VER={str(row['_verifier']):<13} "
            f"JUD={str(row['_judge']):<13} "
            f"HYB_J2A_SON={str(row['_hybrid_j2a']):<13} "
            f"HUM={str(row['_human']):<13} "
            f"title={title!r:.60}\n"
        )
    buf.write("\n")

    with AUDIT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    print(f"Appended J2a forced-sonnet gate result to {AUDIT_LOG}")
    print(buf.getvalue())


if __name__ == "__main__":
    main()
