"""Smoke test: dispatch one prompt via claude -p subprocess and print result."""
import json
import tempfile
import time
from pathlib import Path

from src.judge_via_subagents import dispatch_one

prompt = Path("scripts/prompts/row_01.txt").read_text(encoding="utf-8")
print(f"prompt length: {len(prompt)} chars")
t0 = time.time()
with tempfile.TemporaryDirectory(prefix="judge_smoke_") as td:
    result = dispatch_one(prompt, working_dir=Path(td), timeout_s=180)
elapsed = time.time() - t0
print(f"elapsed: {elapsed:.1f}s")
print("---RESULT---")
print(json.dumps(result, indent=2) if result else "NONE")
