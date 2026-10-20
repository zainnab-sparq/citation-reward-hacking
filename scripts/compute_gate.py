"""Compute the §8 A.5 hand-check gate.

Three label sources:
- HUMAN (your_status): the ground truth (pre-reg §8 A.5)
- VERIFIER (from step0_verifier_labels_reveal.csv): the structured pipeline
- AGENT (parsed from evidence_summary): the 4-agent web-research aid

Three contrasts:
- verifier vs human (THE gate number) — 3-way + binary FAB-vs-not
- agent vs human (calibration of the research aid)
- verifier vs agent (the 42%/60% I reported before compaction; recomputed for audit)
"""
import io
import re
import sys
from pathlib import Path
import pandas as pd

BLIND = Path("data/handcheck/step0_blind_handcheck.csv")
REVEAL = Path("data/handcheck/step0_verifier_labels_reveal.csv")
OUT = Path("data/handcheck/step0_gate_result.txt")


def canon_human(s):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    if s in {"REAL", "FAB", "FABRICATED", "UNV", "UNVERIFIABLE"}:
        return {"FAB": "FABRICATED", "UNV": "UNVERIFIABLE"}.get(s, s)
    return None


def canon_verifier(s):
    if not isinstance(s, str):
        return None
    s = s.strip().upper()
    if s in {"REAL", "REAL_CONFIRMED"}:
        return "REAL"
    if s in {"FABRICATED", "FAB"}:
        return "FABRICATED"
    if s in {"UNVERIFIABLE", "UNV"}:
        return "UNVERIFIABLE"
    return s


_AGENT_FOUND = re.compile(r"^\s*FOUND\b", re.I)
_AGENT_NOT_FOUND = re.compile(r"^\s*NOT\s+FOUND\b", re.I)
_AGENT_AMBIG = re.compile(r"^\s*AMBIGUOUS\b", re.I)


def parse_agent(s):
    """Convert agent evidence_summary to a label.

    Convention from agent-mode-1 prompt:
      'FOUND. ...' -> REAL
      'NOT FOUND. ...' -> FABRICATED
      'AMBIGUOUS. ...' -> UNVERIFIABLE (real work but attribution drift / partial match)
    """
    if not isinstance(s, str):
        return None
    if _AGENT_FOUND.match(s):
        return "REAL"
    if _AGENT_NOT_FOUND.match(s):
        return "FABRICATED"
    if _AGENT_AMBIG.match(s):
        return "UNVERIFIABLE"
    return None


def collapse_binary(label):
    """3-way -> binary: FABRICATED vs not."""
    if label == "FABRICATED":
        return "FAB"
    if label in {"REAL", "UNVERIFIABLE"}:
        return "NOT_FAB"
    return None


def agreement_table(a, b, name_a, name_b):
    paired = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(paired)
    if n == 0:
        return f"{name_a} vs {name_b}: no pairs.\n"
    agree = sum(1 for x, y in paired if x == y)
    pct = 100 * agree / n
    buf = io.StringIO()
    buf.write(f"=== {name_a} vs {name_b} (n={n}) ===\n")
    buf.write(f"3-way agreement: {agree}/{n} = {pct:.1f}%\n\n")

    # Confusion matrix
    labels = ["REAL", "FABRICATED", "UNVERIFIABLE"]
    cm = {(la, lb): 0 for la in labels for lb in labels}
    for x, y in paired:
        if x in labels and y in labels:
            cm[(x, y)] += 1
    buf.write(f"Confusion matrix (rows={name_a}, cols={name_b}):\n")
    buf.write(f"{'':<14}" + "".join(f"{l:<14}" for l in labels) + "\n")
    for la in labels:
        buf.write(f"{la:<14}")
        for lb in labels:
            buf.write(f"{cm[(la, lb)]:<14}")
        buf.write("\n")
    buf.write("\n")

    # Binary collapse
    bin_pairs = [
        (collapse_binary(x), collapse_binary(y)) for x, y in paired
    ]
    bin_pairs = [(x, y) for x, y in bin_pairs if x is not None and y is not None]
    bn = len(bin_pairs)
    bagree = sum(1 for x, y in bin_pairs if x == y)
    bpct = 100 * bagree / bn if bn else 0
    buf.write(f"Binary FAB-vs-NOT agreement: {bagree}/{bn} = {bpct:.1f}%\n")
    return buf.getvalue()


def main():
    df = pd.read_csv(BLIND, encoding="utf-8")
    reveal = pd.read_csv(REVEAL, encoding="utf-8")
    merged = df.merge(reveal, on="row_id", how="left", validate="one_to_one")

    merged["_human"] = merged["your_status"].map(canon_human)
    merged["_verifier"] = merged["verifier_status"].map(canon_verifier)
    merged["_agent"] = merged["evidence_summary"].map(parse_agent)

    n_total = len(merged)
    n_human = merged["_human"].notna().sum()
    n_unparsed_human = (
        merged["your_status"].notna() & merged["_human"].isna()
    ).sum()

    buf = io.StringIO()
    buf.write(f"Total rows: {n_total}\n")
    buf.write(f"Human-labeled (parsed): {n_human}\n")
    buf.write(f"Human-labeled but unparseable: {n_unparsed_human}\n")
    if n_unparsed_human:
        bad = merged.loc[
            merged["your_status"].notna() & merged["_human"].isna(),
            ["presentation_order", "row_id", "your_status"],
        ]
        buf.write("  Unparseable rows:\n")
        buf.write(bad.to_string() + "\n")
    buf.write("\n")

    # Distributions
    buf.write("Human label distribution:\n")
    buf.write(merged["_human"].value_counts(dropna=False).to_string() + "\n\n")
    buf.write("Verifier label distribution:\n")
    buf.write(merged["_verifier"].value_counts(dropna=False).to_string() + "\n\n")
    buf.write("Agent label distribution:\n")
    buf.write(merged["_agent"].value_counts(dropna=False).to_string() + "\n\n")

    # The §8 A.5 gate
    buf.write("########################################################\n")
    buf.write("###  §8 A.5 GATE: structured verifier vs HUMAN       ###\n")
    buf.write("########################################################\n")
    buf.write(
        agreement_table(
            merged["_verifier"].tolist(),
            merged["_human"].tolist(),
            "VERIFIER",
            "HUMAN",
        )
    )
    buf.write("\n")

    # Calibration of agent evidence
    buf.write("########################################################\n")
    buf.write("###  AGENT vs HUMAN (calibrates the research aid)    ###\n")
    buf.write("########################################################\n")
    buf.write(
        agreement_table(
            merged["_agent"].tolist(),
            merged["_human"].tolist(),
            "AGENT",
            "HUMAN",
        )
    )
    buf.write("\n")

    # Recompute the pre-compaction number
    buf.write("########################################################\n")
    buf.write("###  VERIFIER vs AGENT (audit recompute, 42%/60%?)   ###\n")
    buf.write("########################################################\n")
    buf.write(
        agreement_table(
            merged["_verifier"].tolist(),
            merged["_agent"].tolist(),
            "VERIFIER",
            "AGENT",
        )
    )
    buf.write("\n")

    # Per-row disagreement detail for verifier-vs-human
    disagree = merged[
        merged["_human"].notna()
        & merged["_verifier"].notna()
        & (merged["_human"] != merged["_verifier"])
    ]
    buf.write("########################################################\n")
    buf.write(f"###  Verifier-vs-human disagreements (n={len(disagree)})\n")
    buf.write("########################################################\n")
    for _, row in disagree.iterrows():
        title = row["title"] if isinstance(row["title"], str) else row["case_name"]
        buf.write(
            f"  pres#{row['presentation_order']:>2} row{row['row_id']:>3} "
            f"{row['domain']:<8} {row['cell']:<14} "
            f"VER={row['_verifier']:<13} HUM={row['_human']:<13} "
            f"  title={title!r:.80}\n"
        )
        if isinstance(row.get("your_notes"), str) and row["your_notes"].strip():
            buf.write(f"        notes: {row['your_notes'][:200]}\n")
    buf.write("\n")

    OUT.write_text(buf.getvalue(), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
