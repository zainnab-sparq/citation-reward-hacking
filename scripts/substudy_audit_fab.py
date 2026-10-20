"""Emit the auditable record of the sub-study's FABRICATED verdicts.

Writes data/judgments/phase_b_substudy/fab9_audit.csv: one row per
judge-FABRICATED verdict on the verifier-REAL subsample, carrying the citation
key, its (domain, cell) stratum, the judge's reasoning, the evidence URL, and
the search count. This makes the "genuine EXISTENT_WRONG_ID, not judge
false-positive" claim in the pre-reg auditable against the judge's own cited
evidence rather than an unfalsifiable assertion.
"""
import csv
import json
from pathlib import Path

from src.judge_via_subagents import _build_qmap

QMAP = Path("data/questions/frozen_900q_legal_amended.jsonl")
STUDY_DIR = Path("data/judgments/phase_b_substudy")
JUDGMENTS = STUDY_DIR / "judgments.jsonl"
OUT = STUDY_DIR / "fab9_audit.csv"


def main() -> int:
    qmap = _build_qmap(QMAP)
    rows = [json.loads(l) for l in JUDGMENTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    fab = [r for r in rows if r["judge_status"] == "FABRICATED"]
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["custom_id", "domain", "cell", "search_count", "evidence_url", "judge_reasoning"])
        for r in sorted(fab, key=lambda r: r["custom_id"]):
            qid = r["custom_id"].split("__")[0]
            q = qmap[qid]
            w.writerow([r["custom_id"], q["domain"], q["cell"], r["judge_search_count"],
                        r["judge_evidence_url"], r["judge_reasoning"]])
    print(f"wrote {OUT} ({len(fab)} FABRICATED verdicts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
