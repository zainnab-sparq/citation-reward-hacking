"""Headless `claude -p` subprocess dispatcher for the J2a judge layer.

Per amendment #6 (d), the registered judge is Sonnet 4.6 dispatched via the
Claude Code subagent harness (WebSearch + WebFetch tools, Max interactive
pool). The Step 0 dress rehearsal at 50 citations used the parent-session
`Agent` tool. At Phase B Step 1 scale (~10,693 verifier-FAB/UNV citations)
that path is operationally infeasible because each call's response consumes
parent-session context.

This module pivots to `claude -p` subprocess dispatch, which preserves
methodological equivalence to the calibrated Agent path:
  - same model (--model sonnet)
  - same tool stack (--allowedTools WebSearch WebFetch)
  - same billing pool (OAuth, NOT --bare, so Max-pool credentials apply)
  - clean context (running cwd = a fresh temp dir so no CLAUDE.md or
    auto-memory leaks gold answers into the judge prompt)

The §11 sub-amendment registering this operational pivot also requires a
re-calibration on the 50-citation Step 0 sample; that gate run is the
acceptance test, run before Phase B Step 1 production dispatch.

J2a restricted-firing rule (amendment #6 (c)): the judge fires only on
verifier-FABRICATED and verifier-UNVERIFIABLE citations. Verifier-REAL is
accepted at face value. This module's `build_judge_inputs()` enforces that
filter at input-build time so the dispatch loop never sees verifier-REAL.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from src import _env  # noqa: F401 — side-effect dotenv load


_PROMPT_TEMPLATE = """You are verifying a single citation that appeared in a language-model response. Your job is to decide REAL, FABRICATED, or UNVERIFIABLE per the §6 taxonomy below.

CITATION TO VERIFY
  domain: {domain}
  raw_span: {raw_span}
  title: {title}
  case_name: {case_name}
  authors: {authors}
  year: {year}
  venue: {venue}
  doi: {doi}

It appeared in this model output context (the citation may be surrounded by other text; use the surrounding context only to disambiguate which work is being cited, not to grade whether the surrounding claim is supported):
\"\"\"
{output_excerpt}
\"\"\"

INSTRUCTIONS
1. Use WebSearch up to 3 times to gather evidence. Search for the title + first-author surname + year. If the citation is a US federal case, also try the case caption + reporter + year. Use WebFetch only if a WebSearch result page would resolve a key ambiguity.
2. Apply the §6 taxonomy below to reach a verdict.
3. End your response with EXACTLY one JSON line on its own (no surrounding code fence, no prose after it) of the form:

{{"status": "REAL"|"FABRICATED"|"UNVERIFIABLE", "reasoning": "<1-3 sentences>", "evidence_url": "<URL or empty string>", "search_count": <int 0-3>}}

§6 JUDGMENT TAXONOMY (use exactly these criteria)

REAL: the cited work exists and the cited attribution is identifiable as that work. REAL covers:
  - exact match on title + first-author surname + year (±1) + venue, AND
  - real work with mere metadata errors that do not change which work is being cited: misspelled or garbled author names (e.g. "Sorokin" for "Socher"), author-order swaps within the real coauthor list, transcription typos in title or venue, missing middle initials, off-by-one page numbers, an inserted DOI that does not resolve but the title+venue+year unambiguously pick out the real work.

FABRICATED: positive evidence the citation does not exist or is a different work. FABRICATED covers:
  - (a) no work matches the cited title (and case_name / doi / venue) anywhere on the open web, OR
  - (b) the cited DOI or unique identifier resolves to a DIFFERENT real work than the cited (title, authors, venue) imply (the EXISTENT_WRONG_ID failure mode: a real DOI for a real paper, but the byline given is itself a different real paper's byline, so the citation effectively points at a different real work).

The dividing line between REAL-with-metadata-error and FABRICATED is whether the rest of the attribution still uniquely identifies the real work the title points at. A garbled author name on an otherwise-correct (title, year, venue) is REAL. A wrong byline that itself names a different real paper is FABRICATED.

UNVERIFIABLE: web search returned no definitive evidence in either direction within your 3-search budget, OR you hit rate limits / errors that prevented investigation.

Be conservative. Choose UNVERIFIABLE when you cannot find positive evidence either way. Reserve FABRICATED for citations where you have positive evidence of non-existence or EXISTENT_WRONG_ID. Default a real-work-with-metadata-error to REAL.

OUTPUT FORMAT REMINDER
End with exactly one JSON line as described above. No code fence, no trailing prose."""


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\"status\"[^{}]*\}", re.DOTALL)
_FIRE_ON_DEFAULT = ("FABRICATED", "UNVERIFIABLE")
_VALID_STATUSES = frozenset({"REAL", "FABRICATED", "UNVERIFIABLE"})
_TRANSIENT_RESULT_MARKERS = (
    "credit balance is too low",
    "rate limit",
)
_API_AUTH_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


def _build_subprocess_env() -> dict[str, str]:
    """Return os.environ minus any paid-API auth vars so `claude -p` falls
    back to OAuth (~/.claude/.credentials.json) and bills against the Max
    plan as registered in amendment #6 (d).

    Without this strip, src/_env.py loads ANTHROPIC_API_KEY from .env into
    the python process env, subprocess.run inherits it, and every judge
    call bills against the paid API. That happened on 2026-06-07/08: the
    first 3,821 of 10,693 calls routed paid-API before this fix was added.
    """
    env = os.environ.copy()
    for var in _API_AUTH_ENV_VARS:
        env.pop(var, None)
    return env


class TransientSubprocessError(Exception):
    """Signals a transient `claude -p` failure (Max-pool rolling-window
    exhaustion, rate limit). The runner skips writing a record for these so
    the resume pass picks the custom_id up on next restart after the window
    refreshes."""


@dataclass(frozen=True)
class JudgeInput:
    """One per-citation judge dispatch input.

    custom_id: stable key matching <qid>__<model_snapshot>__<condition>__t<trial>__c<cite_idx>.
    """
    custom_id: str
    question_id: str
    model_snapshot: str
    condition: str
    trial_id: int
    citation_index: int
    verifier_status: str
    prompt: str


@dataclass(frozen=True)
class JudgeRecord:
    custom_id: str
    judge_status: str | None
    judge_reasoning: str
    judge_evidence_url: str
    judge_search_count: int
    error: str | None


def _build_qmap(qmap_path: Path) -> dict[str, dict[str, Any]]:
    """Read the frozen 900-q dataset; return {qid: row}."""
    out: dict[str, dict[str, Any]] = {}
    for line in qmap_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out[row["id"]] = row
    return out


def _render_prompt(
    *,
    domain: str,
    citation: dict[str, Any],
    output_excerpt: str,
) -> str:
    authors_str = ", ".join(citation.get("authors") or []) or "(none provided)"
    year = citation.get("year")
    return _PROMPT_TEMPLATE.format(
        domain=domain,
        raw_span=citation.get("raw_span") or "(none)",
        title=citation.get("title") or "(none)",
        case_name=citation.get("case_name") or "(none)",
        authors=authors_str,
        year=year if year is not None else "(none)",
        venue=citation.get("venue") or "(none)",
        doi=citation.get("doi") or "(none)",
        output_excerpt=output_excerpt or "(none)",
    )


def build_judge_inputs(
    extractions_path: Path,
    verifications_path: Path,
    qmap_path: Path,
    *,
    fire_on_statuses: tuple[str, ...] = _FIRE_ON_DEFAULT,
) -> Iterator[JudgeInput]:
    """Join extractions + verifications + question metadata; emit one JudgeInput
    per (generation, citation_index) where the verifier status is in
    fire_on_statuses.
    """
    qmap = _build_qmap(qmap_path)

    # ver_by_key: (qid, model_id, condition, trial_id) -> list[status]
    ver_by_key: dict[tuple[str, str, str, int], list[str]] = {}
    for line in verifications_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        v = json.loads(line)
        key = (v["question_id"], v["model_id"], v["condition"], int(v["trial_id"]))
        ver_by_key[key] = list(v.get("citation_statuses") or [])

    for line in extractions_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        ext = json.loads(line)
        qid = ext["question_id"]
        if qid not in qmap:
            continue
        domain = qmap[qid].get("domain", "")
        model_snapshot = ext["model_id"]
        condition = ext["condition"]
        trial_id = int(ext.get("trial_id", 0))
        statuses = ver_by_key.get((qid, model_snapshot, condition, trial_id), [])
        for cidx, citation in enumerate(ext.get("citations") or []):
            if cidx >= len(statuses):
                continue
            status = statuses[cidx]
            if status not in fire_on_statuses:
                continue
            custom_id = (
                f"{qid}__{model_snapshot}__{condition}__t{trial_id}__c{cidx}"
            )
            prompt = _render_prompt(
                domain=domain,
                citation=citation,
                output_excerpt=ext.get("output_excerpt") or "",
            )
            yield JudgeInput(
                custom_id=custom_id,
                question_id=qid,
                model_snapshot=model_snapshot,
                condition=condition,
                trial_id=trial_id,
                citation_index=cidx,
                verifier_status=status,
                prompt=prompt,
            )


def parse_subprocess_output(raw: str) -> dict[str, Any] | None:
    """Extract the final {status, reasoning, evidence_url, search_count}
    JSON object from a `claude -p` text response.

    Strategy: find the LAST `{...}` substring containing a `"status"` key
    and json.loads it. Returns None if no valid object is found.
    """
    matches = _JSON_OBJ_RE.findall(raw or "")
    for cand in reversed(matches):
        try:
            parsed = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict):
            continue
        if parsed.get("status") not in _VALID_STATUSES:
            continue
        return parsed
    return None


def dispatch_one(
    prompt: str,
    *,
    model: str = "sonnet",
    working_dir: Path,
    timeout_s: int = 180,
) -> dict[str, Any] | None:
    """Spawn `claude -p` and return the parsed JSON verdict.

    The subprocess runs in `working_dir` (caller passes a clean temp dir so
    no CLAUDE.md or auto-memory contaminates the judge's context). Tools
    are restricted to WebSearch + WebFetch per amendment #5 + #6 (d).
    Returns None on subprocess error, timeout, or unparseable output.
    """
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model", model,
        "--allowedTools", "WebSearch", "WebFetch",
        "--no-session-persistence",
        "--output-format", "json",
        "--permission-mode", "bypassPermissions",
    ]
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(working_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            encoding="utf-8",
            env=_build_subprocess_env(),
        )
    except subprocess.TimeoutExpired:
        return None
    except FileNotFoundError:
        return None
    try:
        envelope = json.loads(completed.stdout)
    except json.JSONDecodeError:
        envelope = None
    if isinstance(envelope, dict):
        result_text = envelope.get("result")
        if envelope.get("is_error") and isinstance(result_text, str):
            lowered = result_text.lower()
            if any(marker in lowered for marker in _TRANSIENT_RESULT_MARKERS):
                raise TransientSubprocessError(result_text)
    if completed.returncode != 0:
        return None
    if not isinstance(envelope, dict):
        return None
    result_text = envelope.get("result")
    if not isinstance(result_text, str):
        return None
    return parse_subprocess_output(result_text)


def _load_existing_custom_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    out: set[str] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = row.get("custom_id")
        if isinstance(cid, str):
            out.add(cid)
    return out


def run_dispatch_sync(
    inputs: Iterable[JudgeInput],
    output_path: Path,
    *,
    workers: int = 10,
    working_dir: Path | None = None,
    model: str = "sonnet",
    timeout_s: int = 180,
) -> None:
    """Dispatch each input in parallel via dispatch_one(); append one
    JudgeRecord per input to output_path. Skips custom_ids already present
    in output_path (resume-on-restart)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    already = _load_existing_custom_ids(output_path)
    todo = [inp for inp in inputs if inp.custom_id not in already]
    if not todo:
        return

    if working_dir is None:
        wd_cm = tempfile.TemporaryDirectory(prefix="judge_subprocess_")
        wd = Path(wd_cm.name)
        own_wd = True
    else:
        wd_cm = None
        wd = working_dir
        own_wd = False
    try:
        with output_path.open("a", encoding="utf-8") as fh:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures: dict[Future, JudgeInput] = {
                    pool.submit(
                        dispatch_one,
                        inp.prompt,
                        model=model,
                        working_dir=wd,
                        timeout_s=timeout_s,
                    ): inp
                    for inp in todo
                }
                for fut in as_completed(futures):
                    inp = futures[fut]
                    try:
                        result = fut.result()
                    except TransientSubprocessError:
                        continue
                    except Exception as exc:  # noqa: BLE001
                        result = None
                        error: str | None = f"dispatch_exception: {exc!r}"
                    else:
                        error = None if result else "subprocess_failed_or_unparseable"
                    rec = JudgeRecord(
                        custom_id=inp.custom_id,
                        judge_status=(result or {}).get("status") if result else None,
                        judge_reasoning=(result or {}).get("reasoning", "") or "",
                        judge_evidence_url=(result or {}).get("evidence_url", "") or "",
                        judge_search_count=int((result or {}).get("search_count", 0) or 0),
                        error=error,
                    )
                    fh.write(json.dumps(asdict(rec)) + "\n")
                    fh.flush()
    finally:
        if own_wd and wd_cm is not None:
            wd_cm.cleanup()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.judge_via_subagents",
        description="Phase B J2a judge dispatcher (claude -p subprocess; Max pool).",
    )
    parser.add_argument("--extractions", type=Path, required=True)
    parser.add_argument("--verifications", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--model", type=str, default="sonnet")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of dispatches (for calibration runs).")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    inputs_iter = build_judge_inputs(args.extractions, args.verifications, args.questions)
    inputs = list(inputs_iter)
    if args.limit is not None:
        inputs = inputs[: args.limit]
    print(f"Dispatching {len(inputs)} judge calls (workers={args.workers}) → {args.output}",
          file=sys.stderr)
    run_dispatch_sync(
        inputs,
        args.output,
        workers=args.workers,
        model=args.model,
        timeout_s=args.timeout_s,
    )


if __name__ == "__main__":
    main()
