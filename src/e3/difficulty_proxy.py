"""The E3 item-level difficulty proxy (pre-registration section 6, guardrail (a)).

The registered proxy is "the base model's log-probability of producing any
citation for that item". The base of the probed model (Tulu-3-8B) is
Llama-3.1-8B, and Phase C already holds one Llama-3.1-8B generation per
question per prompt condition, each passed through the same citation extractor
as everything else. The proxy is the smoothed log-rate of "at least one
citation extracted" over those draws:

    log((k + 0.5) / (n + 1))

with k the number of draws that produced a citation and n the number of draws.
The smoothing keeps 0-of-n finite. With four conditions per question the
estimate has five distinct levels, which is coarse for a quantity the
registration calls continuous; the number of draws is written next to every
value so the paper can say exactly that.

Usage:
    python -m src.e3.difficulty_proxy            # writes data/activations/e3_difficulty_proxy.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

# Tulu-3-8B is post-trained from this checkpoint (registry: tulu3_8b stage 0).
BASE_OF_PROBE_MODEL = "meta-llama/Llama-3.1-8B"

EXTRACTIONS = Path("data/extractions/phase_c/extractions.jsonl")
OUT = Path("data/activations/e3_difficulty_proxy.jsonl")

_SMOOTHING_K = 0.5
_SMOOTHING_N = 1.0


@dataclass(frozen=True)
class DifficultyProxy:
    question_id: str
    n_draws: int
    n_with_citation: int
    log_prob_any_citation: float


def _smoothed_log_rate(k: int, n: int) -> float:
    return math.log((k + _SMOOTHING_K) / (n + _SMOOTHING_N))


def build_difficulty_proxy(
    extraction_rows: Iterable[Mapping],
    model_ids: tuple[str, ...] = (BASE_OF_PROBE_MODEL,),
) -> dict[str, DifficultyProxy]:
    """One proxy per question from the named base model's extraction rows."""
    wanted = set(model_ids)
    draws: dict[str, int] = defaultdict(int)
    hits: dict[str, int] = defaultdict(int)
    for row in extraction_rows:
        if row["model_id"] not in wanted:
            continue
        qid = row["question_id"]
        draws[qid] += 1
        if row.get("citations"):
            hits[qid] += 1
    return {
        qid: DifficultyProxy(
            question_id=qid,
            n_draws=n,
            n_with_citation=hits[qid],
            log_prob_any_citation=_smoothed_log_rate(hits[qid], n),
        )
        for qid, n in draws.items()
    }


def write_difficulty_proxy(proxy: Mapping[str, DifficultyProxy], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for qid in sorted(proxy):
            fh.write(json.dumps(asdict(proxy[qid])) + "\n")


def load_difficulty_proxy(path: Path) -> dict[str, float]:
    """question_id -> log_prob_any_citation, which is all the probe needs."""
    out = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                out[row["question_id"]] = float(row["log_prob_any_citation"])
    return out


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractions", default=str(EXTRACTIONS))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--model-id", action="append", default=None,
                        help="base model(s) to pool; default is the registered base of the probed model")
    args = parser.parse_args()

    model_ids = tuple(args.model_id) if args.model_id else (BASE_OF_PROBE_MODEL,)
    proxy = build_difficulty_proxy(_read_jsonl(Path(args.extractions)), model_ids=model_ids)
    write_difficulty_proxy(proxy, Path(args.out))

    n_draws = sorted({p.n_draws for p in proxy.values()})
    levels = sorted({p.n_with_citation for p in proxy.values()})
    print(f"models: {list(model_ids)}")
    print(f"questions: {len(proxy)}  draws per question: {n_draws}  distinct hit counts: {levels}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
