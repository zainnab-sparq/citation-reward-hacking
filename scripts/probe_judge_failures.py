"""Probe the J2a subprocess failure mode.

Picks N failed custom_ids from data/judgments/phase_b/claude-subagent-sonnet.jsonl,
rebuilds each JudgeInput, and re-runs `claude -p` with FULL subprocess capture
(stdout, stderr, returncode, elapsed seconds, timeout flag). Bucketed result
distinguishes timeout vs non-zero exit vs envelope-not-json vs result-not-str
vs no-json-in-result.

Run: conda run -n citation-reward-hacking python -m scripts.probe_judge_failures
"""
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from src.judge_via_subagents import (
    _JSON_OBJ_RE,
    build_judge_inputs,
    parse_subprocess_output,
)


JUDGMENTS = Path("data/judgments/phase_b/claude-subagent-sonnet.jsonl")
EXTRACTIONS = Path("data/extractions/phase_b/extractions.jsonl")
VERIFICATIONS = Path("data/verifications/phase_b/verifications.jsonl")
QUESTIONS = Path("data/questions/frozen_900q_legal_amended.jsonl")

N_PROBE = 8


def load_failed_ids(n: int) -> set[str]:
    ids = []
    for line in JUDGMENTS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("error") == "subprocess_failed_or_unparseable":
            ids.append(row["custom_id"])
            if len(ids) >= n:
                break
    return set(ids)


def dispatch_with_capture(prompt: str, working_dir: Path, timeout_s: int = 300) -> dict[str, Any]:
    """Like dispatch_one but returns a full diagnostic dict instead of just the parsed verdict."""
    cmd = [
        "claude", "-p", prompt,
        "--model", "sonnet",
        "--allowedTools", "WebSearch", "WebFetch",
        "--no-session-persistence",
        "--output-format", "json",
        "--permission-mode", "bypassPermissions",
    ]
    t0 = time.time()
    try:
        completed = subprocess.run(
            cmd, cwd=str(working_dir),
            capture_output=True, text=True,
            timeout=timeout_s, encoding="utf-8",
        )
    except subprocess.TimeoutExpired as e:
        return {
            "stage": "TIMEOUT",
            "elapsed_s": round(time.time() - t0, 1),
            "stdout_len": len((e.stdout or b"").decode("utf-8", "replace")) if e.stdout else 0,
            "stderr_tail": (e.stderr or b"").decode("utf-8", "replace")[-500:] if e.stderr else "",
        }
    elapsed = round(time.time() - t0, 1)

    if completed.returncode != 0:
        return {
            "stage": "NONZERO_EXIT",
            "elapsed_s": elapsed,
            "returncode": completed.returncode,
            "stdout_len": len(completed.stdout or ""),
            "stdout_tail": (completed.stdout or "")[-500:],
            "stderr_tail": (completed.stderr or "")[-500:],
        }

    try:
        envelope = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        return {
            "stage": "ENVELOPE_NOT_JSON",
            "elapsed_s": elapsed,
            "json_error": str(e),
            "stdout_head": (completed.stdout or "")[:500],
            "stdout_tail": (completed.stdout or "")[-500:],
            "stderr_tail": (completed.stderr or "")[-500:],
        }

    if not isinstance(envelope, dict):
        return {"stage": "ENVELOPE_NOT_OBJECT", "elapsed_s": elapsed, "envelope_type": type(envelope).__name__}

    # Diagnose envelope shape
    keys = list(envelope.keys())
    result_text = envelope.get("result")
    is_error = envelope.get("is_error", None)
    subtype = envelope.get("subtype", None)
    num_turns = envelope.get("num_turns", None)
    duration_ms = envelope.get("duration_ms", None)
    duration_api_ms = envelope.get("duration_api_ms", None)
    total_cost_usd = envelope.get("total_cost_usd", None)

    if not isinstance(result_text, str):
        return {
            "stage": "RESULT_NOT_STR",
            "elapsed_s": elapsed,
            "envelope_keys": keys,
            "is_error": is_error,
            "subtype": subtype,
            "result_type": type(result_text).__name__,
            "result_value": str(result_text)[:300],
        }

    parsed = parse_subprocess_output(result_text)
    if parsed is None:
        return {
            "stage": "NO_JSON_IN_RESULT",
            "elapsed_s": elapsed,
            "is_error": is_error,
            "subtype": subtype,
            "num_turns": num_turns,
            "duration_ms": duration_ms,
            "duration_api_ms": duration_api_ms,
            "total_cost_usd": total_cost_usd,
            "result_text_len": len(result_text),
            "result_text_head": result_text[:400],
            "result_text_tail": result_text[-400:],
            "json_regex_match_count": len(_JSON_OBJ_RE.findall(result_text or "")),
        }

    return {
        "stage": "OK",
        "elapsed_s": elapsed,
        "is_error": is_error,
        "subtype": subtype,
        "num_turns": num_turns,
        "parsed_status": parsed.get("status"),
    }


def main():
    failed_ids = load_failed_ids(N_PROBE)
    print(f"Loaded {len(failed_ids)} failed custom_ids to probe", file=sys.stderr)

    inputs_by_id = {}
    for inp in build_judge_inputs(EXTRACTIONS, VERIFICATIONS, QUESTIONS):
        if inp.custom_id in failed_ids:
            inputs_by_id[inp.custom_id] = inp
            if len(inputs_by_id) == len(failed_ids):
                break

    print(f"Rebuilt {len(inputs_by_id)}/{len(failed_ids)} JudgeInputs", file=sys.stderr)

    results = []
    with tempfile.TemporaryDirectory(prefix="probe_judge_") as td:
        wd = Path(td)
        for i, (cid, inp) in enumerate(inputs_by_id.items(), 1):
            print(f"\n[{i}/{len(inputs_by_id)}] {cid}", file=sys.stderr)
            diag = dispatch_with_capture(inp.prompt, wd, timeout_s=300)
            diag["custom_id"] = cid
            diag["verifier_status"] = inp.verifier_status
            diag["prompt_len"] = len(inp.prompt)
            results.append(diag)
            print(f"  stage={diag['stage']} elapsed={diag.get('elapsed_s')}s", file=sys.stderr)
            if diag["stage"] != "OK":
                for k, v in diag.items():
                    if k in ("custom_id", "stage", "elapsed_s", "verifier_status", "prompt_len"):
                        continue
                    if isinstance(v, str) and len(v) > 200:
                        print(f"    {k}: {v[:200]}...", file=sys.stderr)
                    else:
                        print(f"    {k}: {v}", file=sys.stderr)

    Path("data/handcheck").mkdir(exist_ok=True)
    out_path = Path("data/handcheck/probe_judge_failures.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}", file=sys.stderr)

    stages: dict[str, int] = {}
    for r in results:
        stages[r["stage"]] = stages.get(r["stage"], 0) + 1
    print(f"\n=== Stage distribution ===", file=sys.stderr)
    for stage, n in sorted(stages.items(), key=lambda x: -x[1]):
        print(f"  {stage:25s}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
