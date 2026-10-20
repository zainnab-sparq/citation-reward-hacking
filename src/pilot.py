"""Phase A pilot orchestrator.

Runs the 15-question pilot dataset through:
  1. elicit.py — generate completions across (question × condition × model)
  2. citation extraction — pull citation strings out of each completion
  3. verify.py — classify each citation REAL / FABRICATED / UNVERIFIABLE
  4. summary — print fabrication-rate table broken out by model and condition

This is the pilot gate (Phase A in the plan): if the pipeline produces
sensible numbers on 15 questions for ~$0.10 of API spend, we have evidence
the full E1 will work. If it doesn't, we fix here before scaling.

Usage:
    python -m src.pilot --questions data/questions/pilot_15.jsonl \\
                       --model claude-sonnet-4-6 \\
                       --output-dir data/generations/pilot/
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.elicit import (
    AnthropicGenerator,
    GenerationSpec,
    PromptCondition,
    Question,
    generate_batch,
)
from src.verify import VerificationCache, VerificationStatus, verify_citation


CONDITIONS = [
    PromptCondition(
        name="citations_expected",
        template=(
            "{question}\n\n"
            "Please answer with at least two academic citations supporting your claim. "
            "Use standard citation format including authors, year, title, and venue or DOI."
        ),
    ),
    PromptCondition(
        name="neutral",
        template="{question}",
    ),
    PromptCondition(
        name="abstention_permitted",
        template=(
            "{question}\n\n"
            "If you are not certain a specific source exists, say so plainly rather than "
            "guessing — uncertainty is preferred over a wrong citation."
        ),
    ),
    # Verbosity de-confounding control (pre-reg §5/§6; plan §16). Asks for a
    # response length-equivalent to citations_expected WITHOUT mentioning
    # citations or sources, so the citations_expected vs length_matched
    # contrast isolates "asked for citations" from "asked for a longer answer."
    # Secondary de-confounding analysis only; NOT a member of the m=5
    # confirmatory E2 family.
    PromptCondition(
        name="length_matched",
        template=(
            "{question}\n\n"
            "Please provide a thorough, well-developed answer of at least three "
            "substantive paragraphs. Be specific and detailed."
        ),
    ),
]


# Citation extraction: find candidate citation strings in free text.
# We split on sentence boundaries and keep candidates that look citation-shaped:
# either a DOI, a 4-digit year in parens, or a "v." case-name pattern (legal).
_DOI_LIKE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>)]+", re.IGNORECASE)
_YEAR_PAREN = re.compile(r"\((\d{4})[a-z]?\)")
_CASE_LIKE = re.compile(r"\b[A-Z][\w'\-]+\s+v\.\s+[A-Z][\w'\-]+", re.IGNORECASE)
_BRACKETED = re.compile(r"\[(\d+)\]\s*([^\[]+?)(?=\[\d+\]|$)", re.DOTALL)


def extract_citations(text: str) -> list[str]:
    """Pull candidate citation strings out of a free-text response.

    Heuristic: pull any paragraph (split on blank line) or numbered/bulleted
    list item containing a DOI, a (year), or a "v." case-name.

    Citations typically span multiple sentences (authors. (year). title. venue.)
    so we deliberately do NOT split on single sentence boundaries — that would
    fragment a citation into useless pieces.
    """
    candidates: list[str] = []

    # Bracketed references like [1] Smith 2020...
    for _, body in _BRACKETED.findall(text):
        body = body.strip().rstrip(",;").strip()
        if body and (_DOI_LIKE.search(body) or _YEAR_PAREN.search(body) or _CASE_LIKE.search(body)):
            candidates.append(body)

    if candidates:
        return candidates

    # Split on blank lines (paragraph breaks) and on numbered/bulleted list markers
    paragraph_splits = re.split(r"\n\s*\n|\n(?=\s*(?:\d+\.|\*|-)\s)", text)
    for chunk in paragraph_splits:
        chunk = chunk.strip()
        # Strip leading list marker "1. ", "* ", "- "
        chunk = re.sub(r"^\s*(?:\d+\.|\*|-)\s+", "", chunk)
        if not chunk or len(chunk) < 20:
            continue
        if _DOI_LIKE.search(chunk) or _YEAR_PAREN.search(chunk) or _CASE_LIKE.search(chunk):
            candidates.append(chunk)
    return candidates


def load_questions(path: Path) -> list[Question]:
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        questions.append(
            Question(
                id=record["id"],
                domain=record["domain"],
                text=record["text"],
                expected_existence=record["expected_existence"],
                metadata=record.get("metadata", {}),
            )
        )
    return questions


async def run_elicit(
    questions: list[Question],
    model_id: str,
    output_path: Path,
) -> None:
    spec = GenerationSpec(
        model_id=model_id,
        provider="anthropic",
        temperature=1.0,
        max_output_tokens=1024,
    )
    generator = AnthropicGenerator()
    await generate_batch(
        questions=questions,
        conditions=CONDITIONS,
        spec=spec,
        generator=generator,
        output_path=output_path,
        concurrency=3,
    )


async def run_verify(
    generations_path: Path,
    cache_path: Path,
    questions: list[Question],
) -> dict[tuple[str, str], dict]:
    """Verify every extracted citation. Returns dict keyed by (question_id, condition)."""
    cache = VerificationCache(cache_path)
    qmap = {q.id: q for q in questions}
    results: dict[tuple[str, str], dict] = {}
    for line in generations_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        question = qmap[record["question_id"]]
        cites = extract_citations(record["output_text"])
        statuses: list[str] = []
        for cite_text in cites:
            res = await verify_citation(cite_text, domain=question.domain, cache=cache)
            statuses.append(res.status.value)
        results[(record["question_id"], record["condition"])] = {
            "n_citations": len(cites),
            "statuses": statuses,
            "fabricated": sum(1 for s in statuses if s == "FABRICATED"),
            "real": sum(1 for s in statuses if s == "REAL"),
            "unverifiable": sum(1 for s in statuses if s == "UNVERIFIABLE"),
        }
    return results


def summarize(results: dict[tuple[str, str], dict], questions: list[Question]) -> None:
    console = Console()
    qmap = {q.id: q for q in questions}

    # Per-condition aggregate
    per_cond: dict[str, Counter] = defaultdict(Counter)
    per_cond_cite_count: dict[str, int] = defaultdict(int)
    for (qid, cond), stats in results.items():
        per_cond[cond]["fabricated"] += stats["fabricated"]
        per_cond[cond]["real"] += stats["real"]
        per_cond[cond]["unverifiable"] += stats["unverifiable"]
        per_cond_cite_count[cond] += stats["n_citations"]

    table = Table(title="Pilot fabrication rates by prompt condition")
    table.add_column("Condition")
    table.add_column("# Citations")
    table.add_column("Real")
    table.add_column("Fabricated")
    table.add_column("Unverifiable")
    table.add_column("Fabrication rate")
    for cond, counts in per_cond.items():
        n = per_cond_cite_count[cond]
        rate = counts["fabricated"] / n if n else 0.0
        table.add_row(
            cond,
            str(n),
            str(counts["real"]),
            str(counts["fabricated"]),
            str(counts["unverifiable"]),
            f"{rate:.1%}",
        )
    console.print(table)

    # Per-question detail
    detail = Table(title="Per-question detail (answerable vs unanswerable)")
    detail.add_column("Q ID")
    detail.add_column("Domain")
    detail.add_column("Expected")
    for cond in [c.name for c in CONDITIONS]:
        detail.add_column(f"{cond} (real/fab/unv)")
    for qid in sorted(qmap):
        q = qmap[qid]
        row = [qid, q.domain, q.expected_existence]
        for cond in [c.name for c in CONDITIONS]:
            stats = results.get((qid, cond), {})
            row.append(
                f"{stats.get('real', '-')}/{stats.get('fabricated', '-')}/{stats.get('unverifiable', '-')}"
            )
        detail.add_row(*row)
    console.print(detail)


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/questions/pilot_15.jsonl"),
    )
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/generations/pilot"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = args.output_dir / f"{args.model}.jsonl"
    cache_path = Path("data/verifications/pilot_cache.jsonl")

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions from {args.questions}")

    print(f"Eliciting {len(questions) * len(CONDITIONS)} responses from {args.model} ...")
    await run_elicit(questions, args.model, generations_path)

    print("Verifying citations ...")
    results = await run_verify(generations_path, cache_path, questions)

    summarize(results, questions)


if __name__ == "__main__":
    asyncio.run(main())
