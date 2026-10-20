"""Filter medical obscure-but-real candidates against the Retraction Watch DB.

Critic identified a confound: an RLHF-trained model that correctly refuses to
cite a retracted paper would appear as "correct abstention" rather than
"capability-driven recall failure." Filtering retracted papers out of the
medical obscure-real cell removes this artifact.

Source: Retraction Watch DB published by Crossref Labs at
https://api.labs.crossref.org/data/retractionwatch (free CSV download).

Saved snapshot: data/external/retraction_watch_<date>.csv
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_RW_CSV = Path("data/external/retraction_watch_2026-05-03.csv")
_DOI_COLUMN = "OriginalPaperDOI"
_PMID_COLUMN = "OriginalPaperPubMedID"


def load_retracted_dois(csv_path: Path = _RW_CSV) -> set[str]:
    """Load the set of OriginalPaperDOI values, normalized to lowercase."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Retraction Watch CSV not found at {csv_path}. Download from "
            "https://api.labs.crossref.org/data/retractionwatch?email=YOUR_EMAIL"
        )
    dois: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            doi = (row.get(_DOI_COLUMN) or "").strip().lower()
            if doi:
                dois.add(doi)
    return dois


def filter_jsonl(
    input_path: Path,
    output_path: Path,
    rejected_path: Path,
    csv_path: Path = _RW_CSV,
) -> tuple[int, int]:
    """Read candidate JSONL, drop any whose DOI is in Retraction Watch.

    Returns (kept, rejected) counts. Both files are written.
    """
    retracted = load_retracted_dois(csv_path)
    kept = 0
    rejected = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout, \
         rejected_path.open("w", encoding="utf-8") as frej:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            doi = (record.get("expected_citation", {}).get("doi") or "").lower()
            if doi and doi in retracted:
                record.setdefault("metadata", {})["rejected_reason"] = "retraction_watch_match"
                frej.write(json.dumps(record, ensure_ascii=False) + "\n")
                rejected += 1
            else:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
    return kept, rejected


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rejected", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=_RW_CSV)
    args = parser.parse_args()
    kept, rejected = filter_jsonl(args.input, args.output, args.rejected, args.csv)
    print(f"Retraction-watch filter: kept={kept} rejected={rejected}")


if __name__ == "__main__":
    main()
