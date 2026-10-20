"""Dump full claude -p envelope for one failed custom_id."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.judge_via_subagents import build_judge_inputs


TARGET = "medical_known_real_0673__claude-haiku-4-5-20251001__neutral__t0__c2"


def main():
    inp = None
    for x in build_judge_inputs(
        Path("data/extractions/phase_b/extractions.jsonl"),
        Path("data/verifications/phase_b/verifications.jsonl"),
        Path("data/questions/frozen_900q_legal_amended.jsonl"),
    ):
        if x.custom_id == TARGET:
            inp = x
            break
    if inp is None:
        print("NOT FOUND", file=sys.stderr)
        return

    cmd = [
        "claude", "-p", inp.prompt,
        "--model", "sonnet",
        "--allowedTools", "WebSearch", "WebFetch",
        "--no-session-persistence",
        "--output-format", "json",
        "--permission-mode", "bypassPermissions",
    ]
    with tempfile.TemporaryDirectory(prefix="probe_") as td:
        r = subprocess.run(cmd, cwd=td, capture_output=True, text=True, timeout=120, encoding="utf-8")

    print(f"prompt_len: {len(inp.prompt)}")
    print(f"returncode: {r.returncode}")
    print("=== STDOUT ===")
    print(r.stdout)
    print("=== STDERR ===")
    print(r.stderr)

    if r.stdout.strip():
        try:
            env = json.loads(r.stdout)
            print("=== PARSED ENVELOPE KEYS ===")
            for k, v in env.items():
                vstr = str(v)
                if len(vstr) > 200:
                    vstr = vstr[:200] + "..."
                print(f"  {k}: {vstr}")
        except json.JSONDecodeError as e:
            print(f"envelope not parseable: {e}")


if __name__ == "__main__":
    main()
