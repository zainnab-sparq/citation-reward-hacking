"""Build per-citation judge-input JSONL for the 50 Step 0 hand-checked rows.

Joins:
- data/handcheck/step0_blind_handcheck.csv (the human-labeled rows)
- data/extractions/pilot_b/claude-haiku-4-5-20251001.jsonl (for authors)

Output: data/judgments/pilot_b/step0_judge_inputs.jsonl with one record per
citation in the shape src/judge_batch.build_judge_requests expects.
"""
import json
from pathlib import Path
import pandas as pd

HANDCHECK = Path("data/handcheck/step0_blind_handcheck.csv")
EXTRACTIONS = Path("data/extractions/pilot_b/claude-haiku-4-5-20251001.jsonl")
OUT = Path("data/judgments/pilot_b/step0_judge_inputs.jsonl")

MODEL_ID = "claude-haiku-4-5-20251001"
TRIAL_ID = 0  # Step 0 was 1-trial


def main():
    df = pd.read_csv(HANDCHECK, encoding="utf-8")
    extractions = [
        json.loads(line)
        for line in EXTRACTIONS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # Join key: (question_id, condition, citation_index) -> authors
    authors_lookup: dict[tuple[str, str, int], list[str]] = {}
    for ext in extractions:
        qid = ext["question_id"]
        cond = ext["condition"]
        for idx, cite in enumerate(ext.get("citations", [])):
            authors_lookup[(qid, cond, idx)] = cite.get("authors") or []

    OUT.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    missing_authors = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for _, row in df.iterrows():
            qid = row["question_id"]
            cond = row["condition"]
            cidx = int(row["citation_index"])
            authors = authors_lookup.get((qid, cond, cidx), [])
            if not authors:
                missing_authors += 1
            record = {
                "question_id": qid,
                "model_id": MODEL_ID,
                "condition": cond,
                "trial_id": TRIAL_ID,
                "citation_index": cidx,
                "domain": row["domain"],
                "raw_span": row["raw_span"] if isinstance(row["raw_span"], str) else "",
                "title": row["title"] if isinstance(row["title"], str) else None,
                "case_name": row["case_name"] if isinstance(row["case_name"], str) else None,
                "authors": authors,
                "year": int(row["year"]) if pd.notna(row["year"]) else None,
                "venue": row["venue"] if isinstance(row["venue"], str) else None,
                "doi": row["doi"] if isinstance(row["doi"], str) else None,
                "output_excerpt": row["output_excerpt"] if isinstance(row["output_excerpt"], str) else "",
                # Carry the row_id forward so post-judge join to human labels is trivial.
                "_handcheck_row_id": int(row["row_id"]),
            }
            fh.write(json.dumps(record) + "\n")
            written += 1

    print(f"Wrote {written} citation inputs to {OUT}")
    print(f"  rows with authors: {written - missing_authors}")
    print(f"  rows with no authors (case-only / case_name path): {missing_authors}")


if __name__ == "__main__":
    main()
