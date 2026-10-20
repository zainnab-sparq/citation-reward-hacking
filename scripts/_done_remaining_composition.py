"""Report the domain×cell composition of the 3,821 subprocess-judged citations
vs the 6,872 workflow-remaining, for the §11 batch-equivalence sensitivity note."""
import json
import re
from collections import Counter
from pathlib import Path

from src.judge_via_subagents import build_judge_inputs
from src.judge_via_workflow import successful_custom_ids

EXTRACTIONS = Path("data/extractions/phase_b/extractions.jsonl")
VERIFICATIONS = Path("data/verifications/phase_b/verifications.jsonl")
QUESTIONS = Path("data/questions/frozen_900q_legal_amended.jsonl")
OUTPUT = Path("data/judgments/phase_b/claude-subagent-sonnet.jsonl")

_CELL = re.compile(r"^(academic|legal|medical)_(known_real|obscure_real|unanswerable)_")


def cell_of(cid: str) -> str:
    m = _CELL.match(cid)
    return f"{m.group(1)}/{m.group(2)}" if m else "??"


def main():
    done = successful_custom_ids(OUTPUT)
    done_c, rem_c = Counter(), Counter()
    for inp in build_judge_inputs(EXTRACTIONS, VERIFICATIONS, QUESTIONS):
        (done_c if inp.custom_id in done else rem_c)[cell_of(inp.custom_id)] += 1

    keys = sorted(set(done_c) | set(rem_c))
    print(f"{'cell':24s} {'done(subproc)':>14s} {'remaining(wf)':>14s}")
    for k in keys:
        print(f"{k:24s} {done_c[k]:>14d} {rem_c[k]:>14d}")
    print(f"{'TOTAL':24s} {sum(done_c.values()):>14d} {sum(rem_c.values()):>14d}")


if __name__ == "__main__":
    main()
