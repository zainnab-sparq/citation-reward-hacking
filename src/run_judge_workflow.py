"""Orchestration CLI for the Path 7 (Workflow-tool / Max-OAuth) J2a judge batch.

Subcommands
-----------
prep    Build the remaining (un-judged) J2a items, write each prompt to a file,
        and generate one self-contained Workflow script per batch. The batch
        scripts embed their items as a JS literal so the orchestrator passes
        only `scriptPath` (no `args`) — the batch data never transits the
        orchestrator's context.

persist Normalize a completed batch's returned `results` JSON into the canonical
        6-key schema and append the successes to the output JSONL. Failures are
        left un-persisted so the next `prep` re-queues them.

status  Print remaining vs done counts.

The judged content is byte-identical to the registered subprocess batch: each
prompt file is the canonical v2 §6 prompt from build_judge_inputs, written
verbatim, and the workflow agent Reads it. Model is pinned to sonnet and tools
to WebSearch+WebFetch, matching amendment #5 / #6 (d).

Resume safety: every subcommand recomputes the remaining set from the output
JSONL, so a killed run (or an orchestrator-context compaction) is recovered by
re-running `prep` then continuing from the lowest un-run batch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.judge_via_workflow import (
    build_remaining_items,
    chunk,
    make_workflow_args,
    persist_records,
    records_from_workflow_results,
    successful_custom_ids,
)

DEFAULT_CACHE = Path.home() / ".cache" / "citation-judge-prompts"
DEFAULT_EXTRACTIONS = Path("data/extractions/phase_b/extractions.jsonl")
DEFAULT_VERIFICATIONS = Path("data/verifications/phase_b/verifications.jsonl")
DEFAULT_QUESTIONS = Path("data/questions/frozen_900q_legal_amended.jsonl")
DEFAULT_OUTPUT = Path("data/judgments/phase_b/claude-subagent-sonnet.jsonl")
DEFAULT_BATCH_SIZE = 500

# The judge schema + per-item dispatch closure are identical across batches;
# only the embedded `items` literal differs. {name}, {items_json} are filled.
_WORKFLOW_TEMPLATE = """export const meta = {{
  name: {name_json},
  description: 'Path 7 J2a judge batch (Max-OAuth via Workflow agent; Sonnet 4.6 + WebSearch + WebFetch; byte-identical v2 §6 prompt read from file).',
  phases: [
    {{ title: 'Judge', detail: 'Each agent Reads its prompt file, runs WebSearch, returns a §6 verdict' }},
  ],
}}

phase('Judge')

const JUDGE_SCHEMA = {{
  type: 'object',
  additionalProperties: false,
  required: ['status', 'reasoning', 'evidence_url', 'search_count'],
  properties: {{
    status: {{ type: 'string', enum: ['REAL', 'FABRICATED', 'UNVERIFIABLE'] }},
    reasoning: {{ type: 'string', minLength: 1 }},
    evidence_url: {{ type: 'string' }},
    search_count: {{ type: 'integer', minimum: 0, maximum: 3 }},
  }},
}}

const items = {items_json}

log(`J2a batch {name}: dispatching ${{items.length}} judge agents`)

const results = await parallel(items.map((item) => async () => {{
  const taskPrompt = `Read the file at this exact path using the Read tool:\\n${{item.prompt_path}}\\n\\nThat file contains your complete task: a single citation-verification instruction set with a §6 taxonomy. Follow those instructions exactly — use WebSearch (up to 3 times) and WebFetch as needed — and produce the required verdict in the structured output.`
  try {{
    const verdict = await agent(taskPrompt, {{
      label: `judge:${{item.custom_id.slice(0, 28)}}`,
      model: 'sonnet',
      schema: JUDGE_SCHEMA,
    }})
    if (!verdict) return {{ custom_id: item.custom_id, error: 'null_verdict' }}
    return {{ custom_id: item.custom_id, ...verdict }}
  }} catch (e) {{
    return {{ custom_id: item.custom_id, error: String((e && e.message) || e) }}
  }}
}}))

return {{ batch: {name_json}, n: items.length, results }}
"""


def _resolve_paths(args: argparse.Namespace):
    return (
        Path(args.extractions),
        Path(args.verifications),
        Path(args.questions),
        Path(args.output),
        Path(args.cache),
    )


def cmd_prep(args: argparse.Namespace) -> int:
    extractions, verifications, questions, output, cache = _resolve_paths(args)
    prompts_dir = cache / "prompts"
    batches_dir = cache / "batches"
    batches_dir.mkdir(parents=True, exist_ok=True)

    items = build_remaining_items(extractions, verifications, questions, output, prompts_dir)
    print(f"remaining to judge: {len(items)}")
    if not items:
        print("nothing to do — all judged.")
        return 0

    batches = chunk(items, args.batch_size)
    # Clear stale batch scripts so a re-prep after partial completion doesn't
    # leave already-done batches lying around with a mismatched index.
    for old in batches_dir.glob("batch_*.js"):
        old.unlink()

    script_paths: list[str] = []
    for idx, batch in enumerate(batches):
        name = f"j2a-batch-{idx:03d}"
        blob = make_workflow_args(batch)  # {"items": [...]}
        items_json = json.dumps(blob["items"])
        script = _WORKFLOW_TEMPLATE.format(
            name=name,
            name_json=json.dumps(name),
            items_json=items_json,
        )
        path = batches_dir / f"batch_{idx:03d}.js"
        # LF only: the Workflow permission handler rejects a scriptPath whose
        # file carries CR bytes, and write_text would emit CRLF on Windows.
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(script)
        script_paths.append(str(path))

    print(f"wrote {len(batches)} batch scripts (batch_size={args.batch_size}) to {batches_dir}")
    print("\nbatch script paths (run each via Workflow scriptPath, then `persist`):")
    for p in script_paths:
        print(f"  {p}")
    # Machine-readable manifest for the orchestrator loop.
    manifest = batches_dir / "manifest.json"
    manifest.write_text(json.dumps({"batch_size": args.batch_size,
                                     "n_items": len(items),
                                     "scripts": script_paths}, indent=2), encoding="utf-8")
    print(f"\nmanifest: {manifest}")
    return 0


def cmd_persist(args: argparse.Namespace) -> int:
    _, _, _, output, _ = _resolve_paths(args)
    result_path = Path(args.result)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    # Accept three shapes:
    #  (a) the Workflow task-output envelope {summary, agentCount, logs, result:{...,results:[...]}}
    #  (b) the script's bare return {batch, n, results:[...]}
    #  (c) a bare list of result entries
    if isinstance(payload, dict):
        if "results" in payload:
            results = payload["results"]
        elif isinstance(payload.get("result"), dict) and "results" in payload["result"]:
            results = payload["result"]["results"]
        else:
            results = []
    elif isinstance(payload, list):
        results = payload
    else:
        print(f"unrecognized result payload type: {type(payload).__name__}", file=sys.stderr)
        return 2

    records, failed = records_from_workflow_results(results)
    persist_records(records, output)
    print(f"persisted {len(records)} judgments; {len(failed)} failed (will re-queue on next prep)")
    if failed:
        print("failed custom_ids (first 10):")
        for cid in failed[:10]:
            print(f"  {cid}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    extractions, verifications, questions, output, cache = _resolve_paths(args)
    prompts_dir = cache / "prompts"
    done = successful_custom_ids(output)
    # Count remaining WITHOUT rewriting prompt files (dry).
    from src.judge_via_subagents import build_judge_inputs
    total = 0
    remaining = 0
    for inp in build_judge_inputs(extractions, verifications, questions):
        total += 1
        if inp.custom_id not in done:
            remaining += 1
    print(f"total J2a inputs : {total}")
    print(f"done (clean)     : {len(done)}")
    print(f"remaining        : {remaining}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.run_judge_workflow")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("prep", "persist", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--extractions", default=str(DEFAULT_EXTRACTIONS))
        sp.add_argument("--verifications", default=str(DEFAULT_VERIFICATIONS))
        sp.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
        sp.add_argument("--output", default=str(DEFAULT_OUTPUT))
        sp.add_argument("--cache", default=str(DEFAULT_CACHE))
        if name == "prep":
            sp.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
        if name == "persist":
            sp.add_argument("--result", required=True, help="Path to a completed batch's result JSON.")

    args = parser.parse_args(argv)
    if args.command == "prep":
        return cmd_prep(args)
    if args.command == "persist":
        return cmd_persist(args)
    if args.command == "status":
        return cmd_status(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
