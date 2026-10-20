"""Recompute the §8 A.5 gate with the LLM-judge (§11 amendment #5).

Three combined contrasts:
- VERIFIER-ALONE vs HUMAN (the failing baseline, recomputed for audit)
- JUDGE-ALONE vs HUMAN (the ceiling — what a web-augmented judge can do alone)
- HYBRID (verifier-REAL preserved, otherwise judge) vs HUMAN (the deployable Phase B pipeline)

Writes appended sections to data/handcheck/step0_gate_result.txt so the
structured-verifier-alone numbers from compute_gate.py stay alongside the new
hybrid result for the §11 amendment audit trail.
"""
import io
import json
from pathlib import Path

import pandas as pd

BLIND = Path("data/handcheck/step0_blind_handcheck.csv")
REVEAL = Path("data/handcheck/step0_verifier_labels_reveal.csv")
JUDGMENTS = Path("data/judgments/pilot_b/claude-sonnet-4-6.jsonl")
OUT = Path("data/handcheck/step0_gate_result.txt")

_VALID = {"REAL", "FABRICATED", "UNVERIFIABLE"}


def canon_human(s):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    return {"FAB": "FABRICATED", "UNV": "UNVERIFIABLE"}.get(s, s) if s in (
        _VALID | {"FAB", "UNV"}
    ) else None


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


def agreement_block(a, b, name_a, name_b):
    paired = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(paired)
    if n == 0:
        return f"{name_a} vs {name_b}: no pairs.\n"
    agree = sum(1 for x, y in paired if x == y)
    pct = 100 * agree / n
    buf = io.StringIO()
    buf.write(f"=== {name_a} vs {name_b} (n={n}) ===\n")
    buf.write(f"3-way agreement: {agree}/{n} = {pct:.1f}%\n\n")
    labels = ["REAL", "FABRICATED", "UNVERIFIABLE"]
    cm = {(la, lb): 0 for la in labels for lb in labels}
    for x, y in paired:
        if x in labels and y in labels:
            cm[(x, y)] += 1
    buf.write(f"Confusion (rows={name_a}, cols={name_b}):\n")
    buf.write(f"{'':<14}" + "".join(f"{l:<14}" for l in labels) + "\n")
    for la in labels:
        buf.write(f"{la:<14}")
        for lb in labels:
            buf.write(f"{cm[(la, lb)]:<14}")
        buf.write("\n")
    buf.write("\n")
    bin_pairs = [(collapse_binary(x), collapse_binary(y)) for x, y in paired]
    bin_pairs = [(x, y) for x, y in bin_pairs if x is not None and y is not None]
    bn = len(bin_pairs)
    bagree = sum(1 for x, y in bin_pairs if x == y)
    bpct = 100 * bagree / bn if bn else 0
    buf.write(f"Binary FAB-vs-NOT agreement: {bagree}/{bn} = {bpct:.1f}%\n\n")
    return buf.getvalue()


def main():
    df = pd.read_csv(BLIND, encoding="utf-8")
    reveal = pd.read_csv(REVEAL, encoding="utf-8")
    merged = df.merge(reveal, on="row_id", how="left", validate="one_to_one")
    merged["_human"] = merged["your_status"].map(canon_human)
    merged["_verifier"] = merged["verifier_status"].map(canon_verifier)

    judgments = [
        json.loads(line)
        for line in JUDGMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Key the judge by (question_id, condition, citation_index) so we don't
    # rely on the (model_id, trial_id) carry — handcheck CSV doesn't have them.
    judge_by_key = {}
    for j in judgments:
        key = (j["question_id"], j["condition"], int(j["citation_index"]))
        judge_by_key[key] = j

    judge_status = []
    judge_reasoning = []
    judge_evidence_url = []
    judge_search_count = []
    judge_error = []
    for _, row in merged.iterrows():
        key = (row["question_id"], row["condition"], int(row["citation_index"]))
        j = judge_by_key.get(key)
        if j is None:
            judge_status.append(None)
            judge_reasoning.append(None)
            judge_evidence_url.append(None)
            judge_search_count.append(None)
            judge_error.append("missing_judgment")
            continue
        judge_status.append(canon_judge(j.get("judge_status")))
        judge_reasoning.append(j.get("judge_reasoning"))
        judge_evidence_url.append(j.get("judge_evidence_url"))
        judge_search_count.append(j.get("judge_search_count"))
        judge_error.append(j.get("error"))
    merged["_judge"] = judge_status
    merged["_judge_reasoning"] = judge_reasoning
    merged["_judge_evidence_url"] = judge_evidence_url
    merged["_judge_search_count"] = judge_search_count
    merged["_judge_error"] = judge_error

    # Strict-hybrid combine (§11 amendment #5 v2): judge fires on EVERY citation;
    # judge overrides verifier whenever it returns a confident REAL/FABRICATED
    # verdict; verifier is consulted only when the judge could not decide
    # (judge_status=UNVERIFIABLE or judge_error set). This catches the
    # EXISTENT_WRONG_ID failure mode (verifier wrongly-REAL on a real-DOI /
    # wrong-byline citation, e.g. pres#22 HCQ Cochrane) without throwing away
    # verifier signal when the judge hit a search-budget cap (e.g. pres#41).
    def combine(row):
        v = row["_verifier"]
        j = row["_judge"]
        j_err = row["_judge_error"]
        if j is None or j == "UNVERIFIABLE" or j_err:
            return v  # judge unavailable / undecided -> defer to verifier
        return j  # judge committed to REAL or FABRICATED -> judge wins

    merged["_hybrid"] = merged.apply(combine, axis=1)

    buf = io.StringIO()
    buf.write("\n")
    buf.write("#" * 70 + "\n")
    buf.write("###  POST-§11-AMENDMENT-#5 GATE RECOMPUTE WITH PHASE H LLM-JUDGE\n")
    buf.write("###  (Sonnet 4.6 + web_search v2 prompt;\n")
    buf.write("###   strict-hybrid = judge wins on REAL/FAB, else verifier)\n")
    buf.write("#" * 70 + "\n\n")

    buf.write(f"Judgments loaded: {len(judgments)}\n")
    buf.write(f"Rows with judge label: {sum(1 for x in judge_status if x is not None)}\n")
    n_err = sum(1 for e in judge_error if e)
    buf.write(f"Judge error rows: {n_err}\n\n")

    buf.write("Judge label distribution:\n")
    buf.write(merged["_judge"].value_counts(dropna=False).to_string() + "\n\n")
    buf.write("Hybrid label distribution:\n")
    buf.write(merged["_hybrid"].value_counts(dropna=False).to_string() + "\n\n")
    buf.write("Average judge search calls per citation: "
              f"{pd.Series(judge_search_count).dropna().mean():.2f}\n\n")

    buf.write("########################################################\n")
    buf.write("###  §8 A.5 GATE (RECOMPUTED): HYBRID vs HUMAN       ###\n")
    buf.write("########################################################\n")
    buf.write(agreement_block(merged["_hybrid"].tolist(), merged["_human"].tolist(),
                              "HYBRID(ver-REAL|judge)", "HUMAN"))

    buf.write("########################################################\n")
    buf.write("###  CEILING CHECK: JUDGE-ALONE vs HUMAN             ###\n")
    buf.write("########################################################\n")
    buf.write(agreement_block(merged["_judge"].tolist(), merged["_human"].tolist(),
                              "JUDGE", "HUMAN"))

    buf.write("########################################################\n")
    buf.write("###  AUDIT: JUDGE vs VERIFIER (where they disagree)  ###\n")
    buf.write("########################################################\n")
    buf.write(agreement_block(merged["_judge"].tolist(), merged["_verifier"].tolist(),
                              "JUDGE", "VERIFIER"))

    # Hybrid disagreement detail (kept short)
    disagree = merged[
        merged["_human"].notna()
        & merged["_hybrid"].notna()
        & (merged["_hybrid"] != merged["_human"])
    ]
    buf.write("########################################################\n")
    buf.write(f"###  Hybrid-vs-human disagreements (n={len(disagree)})\n")
    buf.write("########################################################\n")
    for _, row in disagree.iterrows():
        title = row["title"] if isinstance(row["title"], str) else (
            row["case_name"] if isinstance(row["case_name"], str) else ""
        )
        buf.write(
            f"  pres#{row['presentation_order']:>2} row{row['row_id']:>3} "
            f"{row['domain']:<8} {row['cell']:<14} "
            f"VER={str(row['_verifier']):<13} "
            f"JUD={str(row['_judge']):<13} "
            f"HYB={str(row['_hybrid']):<13} "
            f"HUM={str(row['_human']):<13} "
            f"  title={title!r:.70}\n"
        )
        notes = row.get("your_notes")
        if isinstance(notes, str) and notes.strip():
            buf.write(f"        human notes: {notes[:200]}\n")
        jr = row.get("_judge_reasoning")
        if isinstance(jr, str) and jr.strip():
            buf.write(f"        judge reasoning: {jr[:200]}\n")
    buf.write("\n")

    # Append, don't overwrite — preserve the original verifier-alone block
    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    print(f"Appended hybrid gate result to {OUT}")

    # Also write a small machine-readable summary so the §11 amendment can
    # be SHA-stamped against deterministic numbers.
    hyb_pairs = [(h, u) for h, u in zip(merged["_hybrid"], merged["_human"]) if h and u]
    n3 = len(hyb_pairs); a3 = sum(1 for h, u in hyb_pairs if h == u)
    bn_pairs = [(collapse_binary(h), collapse_binary(u)) for h, u in hyb_pairs]
    bn = sum(1 for x, y in bn_pairs if x and y); ba = sum(1 for x, y in bn_pairs if x == y)
    summary = {
        "hybrid_3way": {"agree": a3, "n": n3, "pct": round(100 * a3 / n3, 1) if n3 else 0},
        "hybrid_binary": {"agree": ba, "n": bn, "pct": round(100 * ba / bn, 1) if bn else 0},
        "gate_threshold_3way_pct": 90.0,
        "gate_passed": (100 * a3 / n3 if n3 else 0) >= 90.0,
    }
    summary_path = Path("data/handcheck/step0_gate_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
