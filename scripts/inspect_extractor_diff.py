"""Inspect direction of Haiku-vs-Sonnet count mismatches."""
import json
from pathlib import Path

s = json.loads(Path("data/extractions/pilot_b_haiku_extractor/comparison_summary.json").read_text(encoding="utf-8"))
print("=== Count-mismatch examples (Sonnet vs Haiku) ===")
for ex in s["examples_count_mismatch_first_5"]:
    print("---")
    print("key:", ex["key"])
    print(f"Sonnet n={ex['sonnet_n']}:", ex["sonnet_titles"])
    print(f"Haiku  n={ex['haiku_n']}:", ex["haiku_titles"])
print()
print("=== Records with any field disagreement (first 5) ===")
for ex in s["examples_record_with_any_field_disagree_first_8"][:5]:
    print("---")
    print("key:", ex["key"])
    sc = ex["sonnet_cites"]; hc = ex["haiku_cites"]
    print(f"  Sonnet ({len(sc)}):")
    for c in sc:
        print("   -", c.get("title") or c.get("case_name"), "|", c.get("authors"), "|", c.get("year"))
    print(f"  Haiku  ({len(hc)}):")
    for c in hc:
        print("   -", c.get("title") or c.get("case_name"), "|", c.get("authors"), "|", c.get("year"))
