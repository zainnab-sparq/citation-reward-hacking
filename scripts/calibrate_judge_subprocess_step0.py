"""Re-calibrate the J2a judge on the 50 Step 0 prompts via the new
`claude -p` subprocess path (src/judge_via_subagents.dispatch_one).

Reads:    scripts/_subagent_prompts.jsonl  (the 50 pre-built prompts)
Writes:   data/judgments/pilot_b/claude-subagent-sonnet-subprocess.jsonl
          (resume-safe; restart picks up where it left off)

Schema matches data/judgments/pilot_b/claude-subagent-sonnet.jsonl so the
existing scripts/recompute_gate_j2a_sonnet.py works against this output
after a one-line path swap.
"""
import json
from pathlib import Path

from src.judge_via_subagents import JudgeInput, run_dispatch_sync


PROMPTS = Path("scripts/_subagent_prompts.jsonl")
OUTPUT_SUBPROC = Path("data/judgments/pilot_b/claude-subagent-sonnet-subprocess-tmp.jsonl")
OUTPUT_FINAL = Path("data/judgments/pilot_b/claude-subagent-sonnet-subprocess.jsonl")


def main():
    rows = [json.loads(l) for l in PROMPTS.read_text(encoding="utf-8").splitlines() if l.strip()]
    inputs = []
    for r in rows:
        inputs.append(JudgeInput(
            custom_id=f"step0_row_{r['row_id']:02d}",
            question_id=r["question_id"],
            model_snapshot="claude-haiku-4-5-20251001",
            condition=r["condition"],
            trial_id=0,
            citation_index=int(r["citation_index"]),
            verifier_status="N/A",
            prompt=r["prompt"],
        ))
    print(f"Dispatching {len(inputs)} Step 0 prompts via subprocess path...")
    run_dispatch_sync(inputs, OUTPUT_SUBPROC, workers=10, timeout_s=180)

    # Reshape to the schema recompute_gate_j2a_sonnet.py expects:
    # {row_id, judge_status, judge_reasoning, judge_evidence_url, judge_search_count, error}
    row_by_custom = {f"step0_row_{r['row_id']:02d}": r["row_id"] for r in rows}
    OUTPUT_FINAL.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FINAL.open("w", encoding="utf-8") as fh:
        for line in OUTPUT_SUBPROC.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cid = rec["custom_id"]
            if cid not in row_by_custom:
                continue
            out = {
                "row_id": row_by_custom[cid],
                "judge_status": rec.get("judge_status"),
                "judge_reasoning": rec.get("judge_reasoning", ""),
                "judge_evidence_url": rec.get("judge_evidence_url", ""),
                "judge_search_count": rec.get("judge_search_count", 0),
                "error": rec.get("error"),
            }
            fh.write(json.dumps(out) + "\n")
    print(f"Wrote per-row output to {OUTPUT_FINAL}")


if __name__ == "__main__":
    main()
