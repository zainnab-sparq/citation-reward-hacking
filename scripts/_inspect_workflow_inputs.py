"""Inspect the remaining (un-judged) J2a inputs to size the Path 7 workflow run.

Reports:
  - total verifier-FAB/UNV inputs
  - how many already have a successful judgment in the output JSONL
  - how many remain
  - rendered-prompt length distribution (to size the prompt files)
  - output_excerpt length distribution (the only large variable field)
"""
import json
from pathlib import Path

from src.judge_via_subagents import build_judge_inputs

EXTRACTIONS = Path("data/extractions/phase_b/extractions.jsonl")
VERIFICATIONS = Path("data/verifications/phase_b/verifications.jsonl")
QUESTIONS = Path("data/questions/frozen_900q_legal_amended.jsonl")
OUTPUT = Path("data/judgments/phase_b/claude-subagent-sonnet.jsonl")


def _existing_done_custom_ids(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        # "done" == successful judgment (error is None and status present)
        if row.get("error") is None and row.get("judge_status"):
            done.add(row["custom_id"])
    return done


def _pctl(values, p):
    if not values:
        return 0
    s = sorted(values)
    idx = min(len(s) - 1, int(round((p / 100) * (len(s) - 1))))
    return s[idx]


def main():
    done = _existing_done_custom_ids(OUTPUT)
    all_inputs = list(build_judge_inputs(EXTRACTIONS, VERIFICATIONS, QUESTIONS))
    remaining = [i for i in all_inputs if i.custom_id not in done]

    prompt_lens = [len(i.prompt) for i in remaining]

    print(f"total verifier-FAB/UNV inputs : {len(all_inputs)}")
    print(f"already judged (success)      : {len(done)}")
    print(f"remaining to judge            : {len(remaining)}")
    print()
    print("rendered-prompt length (chars):")
    print(f"  min   : {min(prompt_lens) if prompt_lens else 0}")
    print(f"  p50   : {_pctl(prompt_lens, 50)}")
    print(f"  p95   : {_pctl(prompt_lens, 95)}")
    print(f"  p99   : {_pctl(prompt_lens, 99)}")
    print(f"  max   : {max(prompt_lens) if prompt_lens else 0}")
    print(f"  total : {sum(prompt_lens)} chars (~{sum(prompt_lens)/1e6:.1f} MB on disk)")
    print()
    # verifier-status split among remaining
    fab = sum(1 for i in remaining if i.verifier_status == "FABRICATED")
    unv = sum(1 for i in remaining if i.verifier_status == "UNVERIFIABLE")
    print(f"remaining verifier split: FABRICATED={fab}  UNVERIFIABLE={unv}")
    # batch sizing preview
    for bs in (250, 500):
        nb = (len(remaining) + bs - 1) // bs
        print(f"  batch_size={bs}: {nb} workflow runs")


if __name__ == "__main__":
    main()
