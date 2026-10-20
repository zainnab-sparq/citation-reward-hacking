"""Freeze the 900-question dataset.

Reads the 9 with-questions cell files, samples 100 per (domain x cell) with
seed=42, writes the canonical frozen_900q.jsonl + QUESTIONS_HASH.txt +
stratified 50-q subsample (with cell labels stripped from the public version).

Pre-reg §9 commitments:
  - Stratified random 50-q sample, ≈17 per domain × ≈5-6 per (domain × cell)
  - Cell labels stripped from publicly-released version
  - Both file hashes (full + 50-q) committed at freeze

Run:
    .venv/Scripts/python.exe -m src.dataset_construction.freeze
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# 900 questions total = 3 cells x 3 domains x 100 per (domain x cell).
# Sized up from 600 to clear the Phase A.6 power gate for E1's declared
# 10pp interaction effect (a-priori power 86.1% at 900 vs 66.5% at 600).
#
# Sub-pool splits within a (domain x cell), each summing to 100:
#   academic_known_real: 50 canonical + 50 moderate    = 100
#   medical_known_real:  60 Cochrane  + 40 non-Cochrane = 100
#   legal_known_real:    50 SCOTUS    + 50 Circuit      = 100
# Each tuple: (input_filename_stem, target_n, final_domain, final_cell_label)
CELL_SPECS = [
    ("academic_known_real_canonical.with_q",   50, "academic", "known_real"),
    ("academic_known_real_moderate.with_q",    50, "academic", "known_real"),
    ("academic_obscure_real.with_q",          100, "academic", "obscure_real"),
    ("academic_unanswerable.with_q",          100, "academic", "unanswerable"),

    ("medical_known_real_cochrane.with_q",     60, "medical",  "known_real"),
    ("medical_known_real_non_cochrane.with_q", 40, "medical",  "known_real"),
    ("medical_obscure_real_filtered.with_q",  100, "medical",  "obscure_real"),
    ("medical_unanswerable.with_q",           100, "medical",  "unanswerable"),

    ("legal_known_real_scotus.with_q",         50, "legal",    "known_real"),
    ("legal_known_real_circuit.with_q",        50, "legal",    "known_real"),
    ("legal_obscure_real_district.with_q",    100, "legal",    "obscure_real"),
    ("legal_unanswerable.with_q",             100, "legal",    "unanswerable"),
]
# Sum: 50+50 + 100 + 100 + 60+40 + 100 + 100 + 50+50 + 100 + 100 = 900 exactly,
# so no post-sampling trim is needed.

SEED_FULL_SAMPLE = 42
SEED_RELEASE_SUBSAMPLE = 42

# 50-question stratified release subsample
RELEASE_PER_DOMAIN_CELL = {
    ("academic", "known_real"):    6,
    ("academic", "obscure_real"):  6,
    ("academic", "unanswerable"):  5,
    ("medical",  "known_real"):    6,
    ("medical",  "obscure_real"):  5,
    ("medical",  "unanswerable"):  6,
    ("legal",    "known_real"):    6,
    ("legal",    "obscure_real"):  5,
    ("legal",    "unanswerable"):  5,
}  # sum = 50


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        # Drop records the Q-gen pass couldn't produce a question for
        # (titles too garbled or missing authors). The pre-reg requires
        # every dataset row to have a question.
        if not rec.get("question"):
            continue
        out.append(rec)
    return out


def _sample(records: list[dict], n: int, seed: int) -> list[dict]:
    if len(records) <= n:
        return list(records)
    rng = random.Random(seed)
    return rng.sample(records, n)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data/questions/with_q"))
    parser.add_argument("--out", type=Path, default=Path("data/questions/frozen_900q.jsonl"))
    parser.add_argument(
        "--release-out",
        type=Path,
        default=Path("data/questions/release_50q_public.jsonl"),
        help="Public 50-q release with cell labels stripped",
    )
    parser.add_argument(
        "--release-out-with-labels",
        type=Path,
        default=Path("data/questions/release_50q_researcher.jsonl"),
        help="Researcher-access 50-q release with cell labels intact",
    )
    parser.add_argument("--hash-out", type=Path, default=Path("data/questions/QUESTIONS_HASH.txt"))
    args = parser.parse_args()

    all_records: list[dict] = []
    print("Sampling cells...")
    for stem, n_target, domain, cell_label in CELL_SPECS:
        path = args.src / f"{stem}.jsonl"
        records = _read_jsonl(path)
        sample = _sample(records, n_target, SEED_FULL_SAMPLE)
        # Normalize cell label so the dataset only has 3 cells per domain
        for r in sample:
            r["cell"] = cell_label
            r["domain"] = domain
        print(f"  {stem:<45} {len(records):>4} -> {len(sample):>3}")
        all_records.extend(sample)

    # Targets sum to exactly 900, but keep a defensive trim and always sort so
    # ordering (and the sequential re-id below) stays deterministic regardless.
    TARGET_TOTAL = 900
    if len(all_records) > TARGET_TOTAL:
        rng = random.Random(SEED_FULL_SAMPLE + 1)
        rng.shuffle(all_records)
        all_records = all_records[:TARGET_TOTAL]
    all_records.sort(key=lambda r: (r["domain"], r["cell"], r.get("id", "")))

    # Re-id sequentially to keep IDs aligned with the final dataset order
    for i, r in enumerate(all_records):
        r["id"] = f"{r['domain']}_{r['cell']}_{i:04d}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" forces LF on every platform so the SHA-256 below is portable
    # (Windows text mode would otherwise emit CRLF and change the hash).
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in all_records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"\nWrote {len(all_records)} records to {args.out}")

    # SHA-256 hash of canonical-formatted file
    full_hash = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(f"SHA-256 (full)    : {full_hash}")

    # 50-q stratified release subsample
    rng = random.Random(SEED_RELEASE_SUBSAMPLE)
    released: list[dict] = []
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for r in all_records:
        by_cell.setdefault((r["domain"], r["cell"]), []).append(r)
    for (domain, cell), n in RELEASE_PER_DOMAIN_CELL.items():
        pool = by_cell.get((domain, cell), [])
        sample = rng.sample(pool, n) if len(pool) >= n else list(pool)
        released.extend(sample)

    # Researcher-access version (with labels)
    args.release_out_with_labels.parent.mkdir(parents=True, exist_ok=True)
    with args.release_out_with_labels.open("w", encoding="utf-8", newline="\n") as fh:
        for r in released:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    researcher_hash = hashlib.sha256(args.release_out_with_labels.read_bytes()).hexdigest()
    print(f"SHA-256 (50q rsrch): {researcher_hash}")

    # Public version: strip cell + expected_citation + metadata.constructed_via etc
    public_records = []
    for r in released:
        public = {
            "id": r["id"],
            "domain": r["domain"],
            "question": r.get("question"),
        }
        public_records.append(public)
    with args.release_out.open("w", encoding="utf-8", newline="\n") as fh:
        for r in public_records:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    public_hash = hashlib.sha256(args.release_out.read_bytes()).hexdigest()
    print(f"SHA-256 (50q pub) : {public_hash}")

    with args.hash_out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(
            "frozen_900q.jsonl: "                  + full_hash + "\n"
            + "release_50q_researcher.jsonl: "     + researcher_hash + "\n"
            + "release_50q_public.jsonl: "         + public_hash + "\n"
        )
    print(f"Hashes written to {args.hash_out}")


if __name__ == "__main__":
    main()
