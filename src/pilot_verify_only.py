"""Verify-only entry point — for cases where generations were produced
out-of-band (e.g., a Claude Code subagent rather than a direct API call).
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from src.pilot import load_questions, run_verify, summarize


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=Path("data/questions/pilot_15.jsonl"))
    parser.add_argument("--generations", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("data/verifications/pilot_cache.jsonl"))
    args = parser.parse_args()

    questions = load_questions(args.questions)
    print(f"Loaded {len(questions)} questions, verifying {args.generations}")
    results = await run_verify(args.generations, args.cache, questions)
    summarize(results, questions)


if __name__ == "__main__":
    asyncio.run(main())
