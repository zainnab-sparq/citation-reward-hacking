"""Scratch: dump the FABRICATED verdict reasonings from the sub-study so the 9
disagreements can be hand-checked for genuine EXISTENT_WRONG_ID vs judge
false-positive."""
import json
from pathlib import Path

rows = [json.loads(l) for l in
        Path("data/judgments/phase_b_substudy/judgments.jsonl").read_text(encoding="utf-8").splitlines()
        if l.strip()]
fab = [r for r in rows if r["judge_status"] == "FABRICATED"]
print(f"{len(fab)} FABRICATED verdicts:")
for r in fab:
    qid = r["custom_id"].split("__")[0]
    print(f"\n[{qid}]  searches={r['judge_search_count']}")
    print("  reason:", r["judge_reasoning"][:500])
    print("  url:", r["judge_evidence_url"][:140])
