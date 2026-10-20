"""Recompute §8 A.5 gate using subagent (Max-Claude-Code) judge labels.

Mirrors scripts/recompute_gate_with_judge.py but reads from
data/judgments/pilot_b/claude-subagent.jsonl, which is keyed on row_id
(handcheck row_id) rather than (question_id, condition, citation_index).
"""
import io
import json
from pathlib import Path

import pandas as pd

BLIND = Path("data/handcheck/step0_blind_handcheck.csv")
REVEAL = Path("data/handcheck/step0_verifier_labels_reveal.csv")
JUDGMENTS = Path("data/judgments/pilot_b/claude-subagent.jsonl")
OUT = Path("data/handcheck/step0_gate_result.txt")

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
        json.loads(line) for line in JUDGMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    judge_by_row = {int(j["row_id"]): j for j in judgments}

    js, jr, jeu, jsc, jerr = [], [], [], [], []
    for _, row in merged.iterrows():
        j = judge_by_row.get(int(row["row_id"]))
        if j is None:
            js.append(None); jr.append(None); jeu.append(None); jsc.append(None); jerr.append("missing_judgment")
            continue
        js.append(canon_judge(j.get("judge_status")))
        jr.append(j.get("judge_reasoning"))
        jeu.append(j.get("judge_evidence_url"))
        jsc.append(j.get("judge_search_count"))
        jerr.append(j.get("error"))
    merged["_judge"] = js
    merged["_judge_reasoning"] = jr
    merged["_judge_error"] = jerr
    merged["_judge_search_count"] = jsc

    # Strict-hybrid combine (judge wins on REAL/FAB; verifier fallback on judge UNV/error)
    def combine(row):
        v = row["_verifier"]; j = row["_judge"]; je = row["_judge_error"]
        if j is None or j == "UNVERIFIABLE" or je:
            return v
        return j

    merged["_hybrid"] = merged.apply(combine, axis=1)

    buf = io.StringIO()
    buf.write("\n" + "#" * 70 + "\n")
    buf.write("###  GATE RECOMPUTE WITH MAX-CLAUDE-CODE SUBAGENT JUDGE\n")
    buf.write("###  (Sonnet 4.6 via Claude Code Agent dispatch + WebSearch/WebFetch)\n")
    buf.write("###  Strict-hybrid: judge wins on REAL/FAB; verifier fallback on judge UNV/err\n")
    buf.write("#" * 70 + "\n\n")
    buf.write(f"Judgments loaded: {len(judgments)}\n")
    buf.write(f"Rows with judge label: {sum(1 for x in js if x is not None)}\n")
    n_err = sum(1 for e in jerr if e)
    buf.write(f"Judge error rows: {n_err}\n\n")

    buf.write("Judge label distribution:\n")
    buf.write(merged["_judge"].value_counts(dropna=False).to_string() + "\n\n")
    buf.write("Hybrid label distribution:\n")
    buf.write(merged["_hybrid"].value_counts(dropna=False).to_string() + "\n\n")
    valid_sc = pd.Series(jsc).dropna()
    if len(valid_sc):
        buf.write(f"Average judge search calls per citation: {valid_sc.mean():.2f}\n\n")

    buf.write("########################################################\n")
    buf.write("###  §8 A.5 GATE (SUBAGENT): HYBRID vs HUMAN         ###\n")
    buf.write("########################################################\n")
    buf.write(agreement_block(merged["_hybrid"].tolist(), merged["_human"].tolist(),
                              "HYBRID(subagent)", "HUMAN"))
    buf.write("########################################################\n")
    buf.write("###  CEILING: SUBAGENT-JUDGE-ALONE vs HUMAN          ###\n")
    buf.write("########################################################\n")
    buf.write(agreement_block(merged["_judge"].tolist(), merged["_human"].tolist(),
                              "SUBAGENT-JUDGE", "HUMAN"))

    disagree = merged[
        merged["_human"].notna() & merged["_hybrid"].notna()
        & (merged["_hybrid"] != merged["_human"])
    ]
    buf.write("########################################################\n")
    buf.write(f"###  Subagent-hybrid-vs-human disagreements (n={len(disagree)})\n")
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
        jr_text = row.get("_judge_reasoning")
        if isinstance(jr_text, str) and jr_text.strip():
            buf.write(f"        subagent reasoning: {jr_text[:250]}\n")
    buf.write("\n")

    with OUT.open("a", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
    print(f"Appended subagent gate result to {OUT}")

    hyb_pairs = [(h, u) for h, u in zip(merged["_hybrid"], merged["_human"]) if h and u]
    n3 = len(hyb_pairs); a3 = sum(1 for h, u in hyb_pairs if h == u)
    bn_pairs = [(collapse_binary(h), collapse_binary(u)) for h, u in hyb_pairs]
    bn = sum(1 for x, y in bn_pairs if x and y); ba = sum(1 for x, y in bn_pairs if x == y)
    summary = {
        "labeler": "max_claude_code_subagent_judge",
        "hybrid_3way": {"agree": a3, "n": n3, "pct": round(100 * a3 / n3, 1) if n3 else 0},
        "hybrid_binary": {"agree": ba, "n": bn, "pct": round(100 * ba / bn, 1) if bn else 0},
        "gate_threshold_3way_pct": 90.0,
        "gate_passed": (100 * a3 / n3 if n3 else 0) >= 90.0,
    }
    Path("data/handcheck/step0_gate_summary_subagent.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
