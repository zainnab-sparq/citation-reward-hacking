"""Join Phase C generations, extractions and verifications into E3's labeled set.

One row per citation: the generation it came from, its character span inside the
answer, and its verdict. H1.3 asks whether a probe separates REAL citations from
FABRICATED + MISGROUNDED ones, so the label is that binary.

Run with --dry-run to check the join and the span-location rate without loading
a model. The location rate is the number that matters: citations the extractor
reported but which cannot be found in the answer text are dropped, and if that
loss were concentrated in one verdict class it would bias the probe's training
set rather than merely shrinking it. The dry run reports the rate per class for
exactly that reason.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.e3.citation_spans import locate_raw_spans

PROBE_MODEL = "allenai/Llama-3.1-Tulu-3-8B"
GENERATIONS = Path("data/generations/phase_c")
EXTRACTIONS = Path("data/extractions/phase_c/extractions.jsonl")
VERIFICATIONS = Path("data/analysis_inputs/phase_c_final/verifications.jsonl")

# H1.3 separates REAL from (FABRICATED + MISGROUNDED). MISGROUNDED is kept here
# because it is in the registered taxonomy and the E1/E2 code handles it, but
# Phase C's verification output contains none: the realized status vocabulary is
# REAL_CONFIRMED (20,904), FABRICATED (18,382) and UNVERIFIABLE (1,554), so the
# fabrication class collapses to FABRICATED in practice.
FABRICATION_STATUSES = {"FABRICATED", "MISGROUNDED"}
REAL_STATUSES = {"REAL_CONFIRMED"}

# UNVERIFIABLE is deliberately in neither class. It is not a real citation and
# not a demonstrated fabrication, so folding it either way would put citations
# of unknown truth value into a probe trained to separate truth from fabrication.
EXCLUDED_STATUSES = {"UNVERIFIABLE"}


def _key(rec) -> tuple:
    return (rec["question_id"], rec["condition"], rec["model_id"], rec.get("trial_id", 0))


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_generations(model_id: str) -> dict:
    fname = model_id.replace("/", "__") + ".jsonl"
    path = GENERATIONS / fname
    if not path.exists():
        raise FileNotFoundError(f"no generations for {model_id}: {path}")
    return {_key(r): r for r in _read_jsonl(path)}


def load_extractions(model_id: str) -> dict:
    return {
        _key(r): r
        for r in _read_jsonl(EXTRACTIONS)
        if r["model_id"] == model_id
    }


def load_verdicts(model_id: str) -> dict:
    out: dict = {}
    for rec in _read_jsonl(VERIFICATIONS):
        if rec["model_id"] != model_id:
            continue
        out.setdefault(_key(rec), {})[rec["citation_index"]] = rec["status"]
    return out


def build_rows(model_id: str = PROBE_MODEL) -> tuple[list[dict], Counter]:
    gens = load_generations(model_id)
    extractions = load_extractions(model_id)
    verdicts = load_verdicts(model_id)

    stats = Counter()
    rows: list[dict] = []

    for key, ext in extractions.items():
        gen = gens.get(key)
        if gen is None:
            stats["no_generation"] += 1
            continue
        per_index = verdicts.get(key, {})
        citations = ext.get("citations") or []
        spans = locate_raw_spans(gen["output_text"], [c.get("raw_span", "") for c in citations])

        for idx, (citation, span) in enumerate(zip(citations, spans)):
            status = per_index.get(idx)
            if status is None:
                stats["no_verdict"] += 1
                continue
            if status in FABRICATION_STATUSES:
                label = 1
            elif status in REAL_STATUSES:
                label = 0
            else:
                stats[f"skipped_status_{status}"] += 1
                continue

            if span is None:
                stats["unlocated"] += 1
                stats[f"unlocated_label_{label}"] += 1
                continue

            stats["located"] += 1
            stats[f"located_label_{label}"] += 1
            rows.append({
                "question_id": key[0],
                "condition": key[1],
                "trial_id": key[3],
                "citation_index": idx,
                "status": status,
                "label": label,
                "char_start": span[0],
                "char_end": span[1],
            })
    return rows, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=PROBE_MODEL)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", default="data/activations/e3_citation_rows.jsonl")
    args = parser.parse_args()

    rows, stats = build_rows(args.model_id)

    print(f"model            : {args.model_id}")
    print(f"usable citations : {len(rows):,}")
    for key in sorted(stats):
        print(f"  {key:<28} {stats[key]:,}")

    located_0 = stats["located_label_0"]
    located_1 = stats["located_label_1"]
    unloc_0 = stats["unlocated_label_0"]
    unloc_1 = stats["unlocated_label_1"]
    print()
    print("span-location rate by class (a gap here biases the probe set):")
    for name, loc, unloc in (("REAL", located_0, unloc_0),
                             ("FABRICATED+MISGROUNDED", located_1, unloc_1)):
        total = loc + unloc
        rate = loc / total * 100 if total else float("nan")
        print(f"  {name:<24} {loc:,}/{total:,} = {rate:.2f}%")

    if args.dry_run:
        print("\ndry run: nothing written")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"\nwrote {len(rows):,} rows to {out}")


if __name__ == "__main__":
    main()
