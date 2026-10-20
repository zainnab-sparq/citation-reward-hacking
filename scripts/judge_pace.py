"""Drift-free pacing clock for the Path 7 J2a Workflow judge loop.

Spreads the 14 batches across ~24h to minimize the per-rolling-window load on
the Max interactive pool. The clock is anchored on LAUNCH times (not completion
times), so batch runtime variance does not accumulate drift: batch k+1 launches
at launch_time(k) + cadence regardless of how long batch k took.

State lives in a JSON file so the cadence survives orchestrator-context
compaction (resume by re-reading it).

Usage:
  python -m scripts.judge_pace mark          # record "a batch just launched, now"
  python -m scripts.judge_pace check         # -> "READY" or "WAIT <seconds>"
  python -m scripts.judge_pace check --raw   # -> integer seconds remaining (<=0 means ready)
"""
import json
import sys
import time
from pathlib import Path

PACE = Path.home() / ".cache" / "citation-judge-prompts" / "pace.json"

# 14 batches × ~100 min start-to-start ≈ 23h span (batch 0 at T0, batch 13 at
# ~21.7h, finishing well inside the 24h budget). Tunable.
DEFAULT_CADENCE_SECONDS = 6000


def mark() -> None:
    PACE.parent.mkdir(parents=True, exist_ok=True)
    cadence = DEFAULT_CADENCE_SECONDS
    if PACE.exists():
        try:
            cadence = int(json.loads(PACE.read_text()).get("cadence_seconds", cadence))
        except (json.JSONDecodeError, ValueError):
            pass
    now = time.time()
    PACE.write_text(json.dumps({"last_launch_epoch": now, "cadence_seconds": cadence}))
    print(f"marked launch at epoch {now:.0f}; next batch due in {cadence}s (~{cadence/60:.0f} min)")


def _seconds_remaining() -> float:
    if not PACE.exists():
        return 0.0
    try:
        d = json.loads(PACE.read_text())
    except json.JSONDecodeError:
        return 0.0
    due = float(d["last_launch_epoch"]) + float(d.get("cadence_seconds", DEFAULT_CADENCE_SECONDS))
    return due - time.time()


def check(raw: bool = False) -> None:
    rem = _seconds_remaining()
    if raw:
        print(int(rem))
        return
    if rem <= 0:
        print("READY")
    else:
        print(f"WAIT {int(rem)}")


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: judge_pace.py {mark|check [--raw]}", file=sys.stderr)
        return 2
    cmd = argv[0]
    if cmd == "mark":
        mark()
    elif cmd == "check":
        check(raw=("--raw" in argv[1:]))
    else:
        print(f"unknown command: {cmd}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
