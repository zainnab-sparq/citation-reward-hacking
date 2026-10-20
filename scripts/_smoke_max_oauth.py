"""Smoke test the env-strip fix: spawn one `claude -p` call via the updated
src.judge_via_subagents pipeline and inspect the envelope.

Pass criteria for Max OAuth routing:
  - is_error: false
  - total_cost_usd: 0  (Max plan)
  - result: non-empty text containing a valid §6 verdict

Failure criteria (forces fallback to Path 2/3):
  - is_error: true with "credit balance" or "auth" markers
  - total_cost_usd: > 0  (env strip incomplete; key picked up from elsewhere)
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from src.judge_via_subagents import (
    _build_subprocess_env,
    build_judge_inputs,
)


def main():
    # Pick one input we know was previously transient-failed.
    inp = None
    for x in build_judge_inputs(
        Path("data/extractions/phase_b/extractions.jsonl"),
        Path("data/verifications/phase_b/verifications.jsonl"),
        Path("data/questions/frozen_900q_legal_amended.jsonl"),
    ):
        if x.custom_id.startswith("legal_known_real_0320__"):
            inp = x
            break
    if inp is None:
        print("FAIL: could not find smoke input", file=sys.stderr)
        sys.exit(2)

    env = _build_subprocess_env()
    print(f"env vars present: {len(env)}", file=sys.stderr)
    print(f"ANTHROPIC_API_KEY present? {'ANTHROPIC_API_KEY' in env}", file=sys.stderr)
    print(f"ANTHROPIC_AUTH_TOKEN present? {'ANTHROPIC_AUTH_TOKEN' in env}", file=sys.stderr)
    print(f"prompt length: {len(inp.prompt)}", file=sys.stderr)

    cmd = [
        "claude", "-p", inp.prompt,
        "--model", "sonnet",
        "--allowedTools", "WebSearch", "WebFetch",
        "--no-session-persistence",
        "--output-format", "json",
        "--permission-mode", "bypassPermissions",
    ]
    with tempfile.TemporaryDirectory(prefix="smoke_") as td:
        r = subprocess.run(
            cmd, cwd=td, capture_output=True, text=True,
            timeout=300, encoding="utf-8", env=env,
        )

    print(f"\nreturncode: {r.returncode}", file=sys.stderr)

    try:
        envelope = json.loads(r.stdout) if r.stdout else None
    except json.JSONDecodeError:
        envelope = None
        print("=== STDOUT (not parseable JSON) ===")
        print(r.stdout[:2000])
        print("=== STDERR ===")
        print(r.stderr[:2000])
        sys.exit(1)

    if not isinstance(envelope, dict):
        print(f"=== Envelope not a dict: {type(envelope).__name__} ===")
        print(r.stdout[:2000])
        sys.exit(1)

    print("\n=== ENVELOPE SUMMARY ===")
    print(f"  type: {envelope.get('type')}")
    print(f"  subtype: {envelope.get('subtype')}")
    print(f"  is_error: {envelope.get('is_error')}")
    print(f"  api_error_status: {envelope.get('api_error_status')}")
    print(f"  total_cost_usd: {envelope.get('total_cost_usd')}")
    print(f"  duration_ms: {envelope.get('duration_ms')}")
    print(f"  num_turns: {envelope.get('num_turns')}")
    print(f"  service_tier: {envelope.get('usage', {}).get('service_tier')}")
    print(f"  web_search_requests: {envelope.get('usage', {}).get('server_tool_use', {}).get('web_search_requests')}")
    print(f"  result (first 500 chars):")
    print(f"    {str(envelope.get('result'))[:500]}")

    cost = envelope.get("total_cost_usd")
    is_error = envelope.get("is_error")

    print("\n=== VERDICT ===")
    if is_error:
        print("  FAIL: claude -p returned is_error=true. Likely still hitting paid API auth.")
        print(f"  Result text: {envelope.get('result')!r}")
        sys.exit(1)
    elif cost is None or cost > 0:
        print(f"  FAIL: total_cost_usd={cost!r} — env strip did NOT route to Max OAuth.")
        print(f"  claude -p is picking up an API key from somewhere other than the env (settings.json? ~/.claude/?).")
        sys.exit(1)
    else:
        print(f"  PASS: total_cost_usd={cost} — Max OAuth engaged. Safe to resume dispatch.")
        sys.exit(0)


if __name__ == "__main__":
    main()
