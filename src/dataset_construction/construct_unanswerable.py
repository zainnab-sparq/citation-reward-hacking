"""Seeded, deterministic generator for unanswerable-cell candidates (medical mode).

Codifies the unanswerable construction step that plan §6 + recruiter.py reference
but that was never previously committed. Candidates are assembled deterministically
from a curated seed file of real (condition, intervention, venue) facts paired with a
fabricated trial title. The non-existence guarantee is NOT provided here — it comes
downstream from validate_unanswerable.py (Levenshtein + SPECTER cosine ≥ 0.85 against
OpenAlex). This module only produces well-formed candidates for that proof to filter.

Medical pattern (plan §4): real_condition + real_intervention + fabricated_trial.

Run:
    .venv/Scripts/python.exe -m src.dataset_construction.construct_unanswerable \\
        --seeds data/questions/seeds/medical_unanswerable_seeds.jsonl \\
        --out data/questions/raw/medical_unanswerable_topup.jsonl \\
        --start-idx 251 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date
from pathlib import Path

DOMAIN = "medical"
CELL = "unanswerable"
CONSTRUCTED_VIA = "real_condition+real_intervention+fabricated_trial"

# Plausible, diverse surname pool (HalluLens-style real-name fragments). Used only
# to give fabricated citations synthetic authorship; never tied to a real paper.
AUTHOR_SURNAMES = [
    "Patel", "Yamamoto", "Garcia", "Chen", "Nakamura", "Singh", "Müller",
    "Johnson", "Martinez", "Wang", "Kumar", "Okafor", "Rossi", "Andersson",
    "Kowalski", "Ferreira", "Haddad", "Nguyen", "Schmidt", "Ivanova",
    "Tanaka", "Cohen", "Reyes", "Larsson", "Petrov", "Mehta", "Fischer",
    "Santos", "Brandt", "Khan",
]

_RCT_QUESTION_TEMPLATES = [
    "Cite the {year} {venue} trial showing {intervention} {effect} in {population}.",
    "What {year} {venue} RCT demonstrated that {intervention} {effect} in {condition}?",
    "Which {year} {venue} trial established that {intervention} {effect}?",
]

_OBS_QUESTION_TEMPLATES = [
    "Cite the {year} {venue} cohort study showing {intervention} {effect} in {population}.",
    "What {year} {venue} observational study demonstrated that {intervention} {effect} in {condition}?",
    "Which {year} {venue} prospective study established that {intervention} {effect}?",
]


class DuplicateTitleError(ValueError):
    """Raised when two seeds share an identical fabricated title."""


def _venue_for_question(venue: str) -> str:
    # Drop a leading "The " so the question reads "the 2020 Lancet trial",
    # not "the 2020 The Lancet trial". The citation keeps the full venue name.
    return venue[4:] if venue.startswith("The ") else venue


def build_question(seed: dict, rng: random.Random) -> str:
    templates = _RCT_QUESTION_TEMPLATES if seed["study_type"] == "rct" else _OBS_QUESTION_TEMPLATES
    template = rng.choice(templates)
    return template.format(
        year=seed["year"],
        venue=_venue_for_question(seed["venue"]),
        intervention=seed["intervention"],
        effect=seed["effect"],
        population=seed["population"],
        condition=seed["condition"],
    )


def assign_authors(rng: random.Random) -> list[str]:
    k = rng.choice([2, 3])
    return rng.sample(AUTHOR_SURNAMES, k)


def build_candidate(seed: dict, idx: int, rng: random.Random, constructed_at: str) -> dict:
    # Question is built before authors so rng-consumption order is stable.
    question = build_question(seed, rng)
    authors = assign_authors(rng)
    return {
        "id": f"{DOMAIN}_{CELL}_{idx:04d}",
        "domain": DOMAIN,
        "cell": CELL,
        "expected_existence": "unanswerable",
        "question": question,
        "expected_citation": {
            "title": seed["title"],
            "authors": authors,
            "year": seed["year"],
            "venue": seed["venue"],
            "doi": None,
            "openalex_id": None,
            "case_name": None,
            "reporter": None,
        },
        "metadata": {
            "constructed_via": CONSTRUCTED_VIA,
            "constructed_at": constructed_at,
            "real_condition": seed["condition"],
            "real_intervention": seed["intervention"],
            "real_venue": seed["venue"],
            "synthetic_trial_acronym": seed.get("acronym"),
        },
    }


def generate_candidates(
    seeds: list[dict], start_idx: int, seed: int, constructed_at: str
) -> list[dict]:
    rng = random.Random(seed)
    seen_titles: set[str] = set()
    out: list[dict] = []
    for offset, s in enumerate(seeds):
        title = s["title"]
        if title in seen_titles:
            raise DuplicateTitleError(title)
        seen_titles.add(title)
        out.append(build_candidate(s, start_idx + offset, rng, constructed_at))
    return out


def _read_seeds(path: Path) -> list[dict]:
    seeds = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            seeds.append(json.loads(line))
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--start-idx", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--constructed-at", default=date.today().isoformat())
    args = parser.parse_args()

    seeds = _read_seeds(args.seeds)
    records = generate_candidates(seeds, args.start_idx, args.seed, args.constructed_at)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} candidates to {args.out} (ids {records[0]['id']}..{records[-1]['id']})")


if __name__ == "__main__":
    main()
