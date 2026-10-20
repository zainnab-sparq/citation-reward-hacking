"""Judge the verifier-REAL citations that can still change a generation's outcome.

WHY. Amendment #6(c) accepts verifier-`REAL` citations at face value and never
judges them. The blind hand-check of 2026-08-11 found **8 of 20 such citations
were actually fabricated, 40.0% [21.9, 61.3]**, against 4.3% [1.7, 7.0] measured
on Phase B under a registered 8% gate. That bias is toward false-REAL, so the
study currently under-counts fabrication, and its exposure is differential by
condition (citations_expected 26.4% of generations at risk versus
length_matched 14.6%), which pushes the amendment #10 contrast
`citations_expected - length_matched` downward by roughly 8.8pp against bin
boundaries of +/-5pp. That is large enough to manufacture the registered
"loss" verdict on its own, which is why this is fixed rather than disclosed.

WHY IT IS CHEAPER THAN IT LOOKS. The registered outcome is generation-level, so
a missed fabrication only changes anything when it is the ONLY bad citation in
its generation. Generations already scored fabricated=1 cannot flip, and
generations holding no verifier-REAL citation have no exposure. Only the
verifier-REAL citations inside AT-RISK generations need judging: about 6,784 of
8,860, roughly 14 nights instead of 18. This does not estimate the bias, it
eliminates it.

WHAT IT DOES NOT DO. It leaves the registered pipeline untouched. Verdicts go to
their own file, `data/judgments/phase_c/verifier_real_rejudge.jsonl`, and
applying them is an explicit, auditable, reversible step in the analysis rather
than a silent edit of the main checkpoint. This is a deviation from amendment
#6(c) and belongs in the §11 sub-amendment that amendment's own >8% rule
triggers.

Prompts are rendered by `build_judge_inputs` with `fire_on_statuses=("REAL",)`,
so they come off the identical renderer the main judge used. The instrument is
the same; only the population differs.

TIMING. The at-risk set is NOT final while the main judge is still running: an
unjudged verifier-FABRICATED citation may yet be rescued to REAL, newly exposing
its generation. `prep` reports how many such citations remain and warns. Re-run
`prep` after the main judge completes; it is idempotent.

Usage:
  python -m src.analysis.verifier_real_rejudge prep \
      --extractions data/extractions/phase_c/extractions.jsonl \
      --verifications data/verifications/phase_c/verifications.jsonl \
      --questions data/questions/frozen_900q_legal_amended.jsonl \
      --judgments data/judgments/phase_c/claude-subagent-sonnet.jsonl \
      --output data/judgments/phase_c/verifier_real_rejudge.jsonl \
      --cache ~/.cache/citation-verifier-real-rejudge

Then run each batch through the Workflow tool and persist with
`src.run_judge_workflow persist --output <the rejudge file>`.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from src.judge_via_subagents import build_judge_inputs
from src.judge_via_workflow import _prompt_filename, successful_custom_ids, chunk, make_workflow_args
from src.run_judge_workflow import _WORKFLOW_TEMPLATE

BAD_STATUSES = ("FABRICATED", "MISGROUNDED")
_JUDGE_WINS = ("REAL", "FABRICATED")
DEFAULT_BATCH_SIZE = 500


def combined_status(verifier_status: str, judge_status: str | None) -> str:
    """The amendment #6(c) combine rule, as used by the analysis."""
    if verifier_status == "REAL":
        return "REAL"
    if judge_status in _JUDGE_WINS:
        return judge_status
    return verifier_status


def _gen_key(record: dict[str, Any]) -> tuple[str, str, str, int]:
    return (record["question_id"], record["condition"], record["model_id"],
            int(record.get("trial_id", 0)))


def _judge_index(judgments: Iterable[dict[str, Any]]) -> dict[str, str]:
    return {
        j["custom_id"]: j.get("judge_status")
        for j in judgments
        if j.get("error") is None and j.get("judge_status")
    }


def at_risk_generation_keys(
    verifications: Iterable[dict[str, Any]],
    judgments: Iterable[dict[str, Any]],
) -> set[tuple[str, str, str, int]]:
    """Generations currently scored fabricated=0 that hold >=1 verifier-REAL citation.

    These are exactly the generations whose outcome the verifier-REAL trust step
    can still be wrong about.
    """
    judged = _judge_index(judgments)
    at_risk: set[tuple[str, str, str, int]] = set()
    for verification in verifications:
        qid, condition, model_id, trial_id = _gen_key(verification)
        n_bad = n_trusted = 0
        for index, verifier_status in enumerate(verification.get("citation_statuses") or []):
            if verifier_status == "REAL":
                n_trusted += 1
                continue
            custom_id = f"{qid}__{model_id}__{condition}__t{trial_id}__c{index}"
            if combined_status(verifier_status, judged.get(custom_id)) in BAD_STATUSES:
                n_bad += 1
        if n_bad == 0 and n_trusted > 0:
            at_risk.add((qid, condition, model_id, trial_id))
    return at_risk


def unjudged_pending_count(
    verifications: Iterable[dict[str, Any]],
    judgments: Iterable[dict[str, Any]],
) -> int:
    """Verifier-FAB/UNV citations with no verdict yet.

    While this is above zero the at-risk set can still grow, because any of them
    could be rescued to REAL and leave its generation newly exposed.
    """
    judged = _judge_index(judgments)
    pending = 0
    for verification in verifications:
        qid, condition, model_id, trial_id = _gen_key(verification)
        for index, verifier_status in enumerate(verification.get("citation_statuses") or []):
            if verifier_status == "REAL":
                continue
            if f"{qid}__{model_id}__{condition}__t{trial_id}__c{index}" not in judged:
                pending += 1
    return pending


def _key_from_custom_id(custom_id: str) -> tuple[str, str, str, int]:
    question_id, model_id, condition, trial, _ = custom_id.split("__")
    return (question_id, condition, model_id, int(trial[1:]))


def select_targets(
    inputs: Iterable[Any],
    at_risk: set[tuple[str, str, str, int]],
    done: set[str],
) -> list[Any]:
    return [
        i for i in inputs
        if i.custom_id not in done and _key_from_custom_id(i.custom_id) in at_risk
    ]


def write_batch_scripts(
    targets: Sequence[Any], cache: Path, batch_size: int = DEFAULT_BATCH_SIZE
) -> list[str]:
    """Write one prompt file per target and one self-contained batch script per chunk."""
    cache = Path(cache)
    prompts_dir = cache / "prompts"
    batches_dir = cache / "batches"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)
    for old in batches_dir.glob("batch_*.js"):
        old.unlink()

    items = []
    for target in targets:
        path = prompts_dir / _prompt_filename(target.custom_id)
        if not path.exists():
            path.write_text(target.prompt, encoding="utf-8", newline="\n")
        items.append(_Item(target.custom_id, str(path)))

    written: list[str] = []
    for index, batch in enumerate(chunk(items, batch_size)):
        name = f"vrr-batch-{index:03d}"
        script = _WORKFLOW_TEMPLATE.format(
            name=name,
            name_json=json.dumps(name),
            items_json=json.dumps(make_workflow_args(batch)["items"]),
        )
        path = batches_dir / f"batch_{index:03d}.js"
        # newline="\n" so the Workflow permission handler never sees a CR.
        path.write_text(script, encoding="utf-8", newline="\n")
        written.append(str(path))
    return written


class _Item:
    __slots__ = ("custom_id", "prompt_path")

    def __init__(self, custom_id: str, prompt_path: str) -> None:
        self.custom_id = custom_id
        self.prompt_path = prompt_path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.analysis.verifier_real_rejudge")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prep", "status"):
        sp = sub.add_parser(name)
        sp.add_argument("--extractions", required=True, type=Path)
        sp.add_argument("--verifications", required=True, type=Path)
        sp.add_argument("--questions", required=True, type=Path)
        sp.add_argument("--judgments", required=True, type=Path)
        sp.add_argument("--output", required=True, type=Path)
        sp.add_argument("--cache", required=True, type=Path)
        if name == "prep":
            sp.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args(argv)

    if args.output.resolve() == args.judgments.resolve():
        parser.error("--output must differ from --judgments: the registered "
                     "checkpoint stays untouched so applying this correction "
                     "remains an explicit, reversible analysis step.")

    verifications = _read_jsonl(args.verifications)
    judgments = _read_jsonl(args.judgments)
    at_risk = at_risk_generation_keys(verifications, judgments)
    pending = unjudged_pending_count(verifications, judgments)
    done = successful_custom_ids(args.output) if args.output.exists() else set()

    inputs = list(build_judge_inputs(args.extractions, args.verifications,
                                     args.questions, fire_on_statuses=("REAL",)))
    targets = select_targets(inputs, at_risk, done)

    print(f"at-risk generations            : {len(at_risk)}")
    print(f"verifier-REAL citations total  : {len(inputs)}")
    print(f"  inside at-risk generations   : "
          f"{sum(1 for i in inputs if _key_from_custom_id(i.custom_id) in at_risk)}")
    print(f"  already re-judged            : {len(done)}")
    print(f"  remaining to judge           : {len(targets)}")

    if pending:
        print(f"\n⚠  {pending} verifier-FAB/UNV citations still have no verdict from the "
              f"main judge.\n   The at-risk set can still GROW (an unjudged citation "
              f"rescued to REAL newly\n   exposes its generation), so re-run prep after "
              f"the main judge completes.")

    if args.command == "status":
        return 0
    if not targets:
        print("\nnothing to do.")
        return 0

    paths = write_batch_scripts(targets, args.cache, args.batch_size)
    print(f"\nwrote {len(paths)} batch scripts to {Path(args.cache) / 'batches'}")
    print(f"persist each completed batch with:\n"
          f"  python -m src.run_judge_workflow persist --result <.output> "
          f"--output {args.output} --cache {args.cache} "
          f"--extractions {args.extractions} --verifications {args.verifications} "
          f"--questions {args.questions}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
