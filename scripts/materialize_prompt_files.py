"""Materialize per-citation prompt files at scripts/prompts/row_NN.txt."""
import json
from pathlib import Path

rows = [
    json.loads(line)
    for line in Path("scripts/_subagent_prompts.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
out_dir = Path("scripts/prompts")
out_dir.mkdir(exist_ok=True)
for r in rows:
    fp = out_dir / f"row_{r['row_id']:02d}.txt"
    fp.write_text(r["prompt"], encoding="utf-8")
print(f"Wrote {len(rows)} prompt files to {out_dir}")
print("First 10 row_ids:", [r["row_id"] for r in rows[:10]])
