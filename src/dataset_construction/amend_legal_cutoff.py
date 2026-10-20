"""Amendment 2026-05-25 (round-3 panel): enforce the pre-2023 cutoff on the legal known-real cell.

The frozen legal known-real cell (frozen_900q.jsonl) was recruited under a
``citeCount >= 50`` Circuit/SCOTUS filter with NO publication-year gate, so it
contained 45 opinions dated >= 2023 (24 from 2023, 16 from 2024, 5 from 2025).
That violates the pre-registered "pre-2023 for known-real and obscure-but-real
cells" restriction (intersection of model training cutoffs) and introduces a
recency/training-cutoff confound into what is supposed to be the capability-ceiling
control cell.

This script replaces each >= 2023 legal known-real item with a same-tier
(SCOTUS <-> SCOTUS, Circuit <-> Circuit) pre-2023 federal opinion drawn from the
same pre-recruited candidate pool under the original recruitment filter
(``cited_by_count >= 50``), selected by a seeded (42) random draw over the unused
pre-2023 candidates and excluding any case already present in the frozen set. Slot
IDs are preserved; the original content is retained under
``metadata.original_expected_citation`` for full provenance.

Pre-registration integrity (pre-reg section 8 amendment procedure):
- The original ``frozen_900q.jsonl`` and its SHA-256 are retained immutably.
- The amended set is written as ``frozen_900q_legal_amended.jsonl`` with its own
  SHA-256. The amended set is the operative dataset for all reported analyses.
- The release-50 subsample is preserved by ID; only the one affected slot
  (legal_known_real_0354) has its content swapped, so sample membership is
  unchanged.

Run: ``python -m src.dataset_construction.amend_legal_cutoff`` from the repo root.
Deterministic: re-running reproduces byte-identical output.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

SEED = 42
CUTOFF_YEAR = 2023  # items with decision year >= this are replaced
MIN_CITE_COUNT = 50  # original recruitment filter for legal known-real
AMENDMENT_TAG = "2026-05-25-r3-pre2023-cutoff"

QDIR = Path("data/questions")
FROZEN = QDIR / "frozen_900q.jsonl"
SCOTUS_POOL = QDIR / "raw" / "legal_known_real_scotus.with_q.jsonl"
CIRCUIT_POOL = QDIR / "raw" / "legal_known_real_circuit.with_q.jsonl"
REL_RESEARCHER = QDIR / "release_50q_researcher.jsonl"
REL_PUBLIC = QDIR / "release_50q_public.jsonl"

OUT_FROZEN = QDIR / "frozen_900q_legal_amended.jsonl"
OUT_REL_RESEARCHER = QDIR / "release_50q_researcher_amended.jsonl"
OUT_REL_PUBLIC = QDIR / "release_50q_public_amended.jsonl"


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def _decision_year(row: dict) -> int | None:
    ec = row.get("expected_citation") or {}
    year = ec.get("year") if isinstance(ec, dict) else None
    return int(year) if year else None


def _tier(row: dict) -> str:
    ec = row.get("expected_citation") or {}
    return "scotus" if (ec.get("court_id") or "").lower() == "scotus" else "circuit"


def _cite_count(row: dict) -> int:
    ec = row.get("expected_citation") or {}
    return int(ec.get("cited_by_count") or 0)


def _usable_spares(pool_path: Path, frozen_questions: set[str]) -> list[dict]:
    spares = []
    for row in _load(pool_path):
        year = _decision_year(row)
        if year is None or year >= CUTOFF_YEAR:
            continue
        if _cite_count(row) < MIN_CITE_COUNT:
            continue
        if row.get("question", "").strip() in frozen_questions:
            continue
        spares.append(row)
    spares.sort(key=lambda r: r["id"])  # stable base order before shuffle
    return spares


def _build_replacement_map(frozen: list[dict]) -> dict[str, dict]:
    """slot_id -> chosen pre-2023 spare, deterministic under SEED."""
    legal_kr = [r for r in frozen if r["domain"] == "legal" and r["cell"] == "known_real"]
    frozen_questions = {r["question"].strip() for r in frozen}

    to_replace = sorted(
        [r for r in legal_kr if (_decision_year(r) or 0) >= CUTOFF_YEAR],
        key=lambda r: r["id"],
    )
    spares = {
        "scotus": _usable_spares(SCOTUS_POOL, frozen_questions),
        "circuit": _usable_spares(CIRCUIT_POOL, frozen_questions),
    }
    rng = random.Random(SEED)
    for pool in spares.values():
        rng.shuffle(pool)

    iters = {tier: iter(pool) for tier, pool in spares.items()}
    used_cases: set[str] = set()
    mapping: dict[str, dict] = {}
    for slot in to_replace:
        tier = _tier(slot)
        spare = next(iters[tier])
        while spare["question"].strip() in used_cases:
            spare = next(iters[tier])
        used_cases.add(spare["question"].strip())
        mapping[slot["id"]] = spare
    return mapping


def _amend_row(slot: dict, spare: dict) -> dict:
    """Swap spare content into the frozen slot, preserving id/domain/cell + provenance."""
    spare_meta = dict(spare.get("metadata") or {})
    spare_meta.update(
        {
            "amended": True,
            "amendment": AMENDMENT_TAG,
            "amendment_reason": "pre-2023 cutoff enforcement on legal known-real cell",
            "original_expected_citation": slot.get("expected_citation"),
            "original_year": _decision_year(slot),
            "replacement_source_id": spare["id"],
        }
    )
    return {
        "id": slot["id"],
        "domain": slot["domain"],
        "cell": slot["cell"],
        "question": spare["question"],
        "expected_existence": slot.get("expected_existence", "answerable"),
        "expected_citation": spare["expected_citation"],
        "metadata": spare_meta,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    frozen = _load(FROZEN)
    mapping = _build_replacement_map(frozen)

    amended = [
        _amend_row(row, mapping[row["id"]]) if row["id"] in mapping else row
        for row in frozen
    ]

    # --- invariants (self-verifying migration) ---
    legal_kr = [r for r in amended if r["domain"] == "legal" and r["cell"] == "known_real"]
    assert len(legal_kr) == 100, f"legal known-real count changed: {len(legal_kr)}"
    assert len(amended) == 900, f"row count changed: {len(amended)}"
    bad = [r["id"] for r in legal_kr if (_decision_year(r) or 0) >= CUTOFF_YEAR]
    assert not bad, f"post-cutoff items remain: {bad}"
    questions = [r["question"].strip() for r in legal_kr]
    assert len(set(questions)) == len(questions), "duplicate legal known-real questions introduced"
    assert all(_cite_count(r) >= MIN_CITE_COUNT for r in legal_kr), "a legal known-real item fell below citeCount floor"

    _write_jsonl(OUT_FROZEN, amended)

    # --- release subsamples: preserve membership, swap only affected slots ---
    amended_by_id = {r["id"]: r for r in amended}
    researcher = _load(REL_RESEARCHER)
    researcher_amended = [amended_by_id.get(r["id"], r) for r in researcher]
    _write_jsonl(OUT_REL_RESEARCHER, researcher_amended)

    public = _load(REL_PUBLIC)
    public_amended = []
    for r in public:
        repl = amended_by_id.get(r["id"])
        if repl is not None and r["id"] in mapping:
            public_amended.append({"domain": r["domain"], "id": r["id"], "question": repl["question"]})
        else:
            public_amended.append(r)
    _write_jsonl(OUT_REL_PUBLIC, public_amended)

    # --- report ---
    n_by_tier = Counter(_tier(r) for r in frozen
                        if r["domain"] == "legal" and r["cell"] == "known_real"
                        and (_decision_year(r) or 0) >= CUTOFF_YEAR)
    print(f"Replaced {len(mapping)} legal known-real items "
          f"(circuit={n_by_tier['circuit']}, scotus={n_by_tier['scotus']}).")
    repl_years = Counter(_decision_year(s) for s in mapping.values())
    print(f"Replacement decision-year distribution: {dict(sorted(repl_years.items()))}")
    print("\nSHA-256 (append to QUESTIONS_HASH.txt, keep originals immutable):")
    for path in (OUT_FROZEN, OUT_REL_RESEARCHER, OUT_REL_PUBLIC):
        print(f"{path.name}: {_sha256(path)}")


if __name__ == "__main__":
    main()
