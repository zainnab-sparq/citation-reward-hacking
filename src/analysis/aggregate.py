"""Phase B aggregation: JSONL artifacts -> R analysis frames.

Reads the four Phase B JSONL streams (generations, extractions [implicit
via verifications.n_citations], verifications, confidences) plus the
frozen question set + model registry, and writes four CSVs whose
schemas match `build_e2_data()` in src/analysis/e2_prompt_manipulation.R:

  questions.csv      id, cell, domain
  models.csv         model_id, family, is_post_trained, is_open_weights
  generations.csv    question_id, condition, model_id, trial_id, output_text
  verifications.csv  question_id, condition, model_id, trial_id,
                     citation_index, status, confidence

Status mapping: the Python verification taxonomy is
{REAL, FABRICATED, UNVERIFIABLE}; the R aggregator's GOOD_STATUSES
expects REAL_CONFIRMED. We map REAL -> REAL_CONFIRMED at flatten time so
the existing R analysis consumes Phase B output unchanged. MISGROUNDED
is Phase H territory and never appears in Phase B output.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


_KeyTuple = tuple[str, str, str, int]
_CitationKey = tuple[str, str, str, int, int]  # + citation_index
_STATUS_MAP = {"REAL": "REAL_CONFIRMED"}  # FABRICATED/UNVERIFIABLE pass through

# R taxonomy buckets (mirror e2_prompt_manipulation.R GOOD_STATUSES / bad set).
_GOOD_STATUSES = {"REAL_CONFIRMED", "EXISTENT_METADATA_ERROR", "EXISTENT_WRONG_ID"}
_BAD_STATUSES = {"FABRICATED", "MISGROUNDED"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        rows.append(json.loads(raw))
    return rows


def flatten_verifications(
    verifications: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Explode one verification record (n citations) into n one-row entries.

    Each output row carries (question_id, condition, model_id, trial_id,
    citation_index, status). REAL is mapped to REAL_CONFIRMED so the R
    GOOD_STATUSES check succeeds.
    """
    for record in verifications:
        for idx, status in enumerate(record.get("citation_statuses", []) or []):
            yield {
                "question_id": record["question_id"],
                "condition": record["condition"],
                "model_id": record["model_id"],
                "trial_id": int(record.get("trial_id", 0)),
                "citation_index": idx,
                "status": _STATUS_MAP.get(status, status),
            }


def to_questions_frame(path: Path) -> list[dict[str, Any]]:
    """Read frozen 900q JSONL into [{id, cell, domain}, ...]."""
    out: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        out.append(
            {
                "id": record["id"],
                "cell": record["cell"],
                "domain": record["domain"],
            }
        )
    return out


def to_models_frame(path: Path) -> list[dict[str, Any]]:
    """Read the Phase B model registry; emit columns the R merge needs."""
    out: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        out.append(
            {
                "model_id": record["model_id"],
                "family": record["family"],
                "is_post_trained": int(record["is_post_trained"]),
                "is_open_weights": int(record["is_open_weights"]),
            }
        )
    return out


def to_generations_frame(path: Path) -> list[dict[str, Any]]:
    """Read generations.jsonl, drop error rows, project to R-merge columns.

    output_text is preserved so `build_e2_data()` can derive output_chars
    for the verbosity-covariate sensitivity (§13).
    """
    out: list[dict[str, Any]] = []
    for record in _read_jsonl(path):
        if record.get("error"):
            continue
        out.append(
            {
                "question_id": record["question_id"],
                "condition": record["condition"],
                "model_id": record["model_id"],
                "trial_id": int(record.get("trial_id", 0)),
                "output_text": record.get("output_text", ""),
            }
        )
    return out


def attach_confidences(
    verification_rows: Iterable[dict[str, Any]],
    confidence_records: Iterable[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Join per-citation confidence onto per-citation verification rows.

    Confidence is matched by (question_id, condition, model_id, trial_id,
    citation_index). Missing confidence (zero-citation generation, errored
    follow-up batch, or an index Sonnet didn't return) is recorded as None
    so the downstream CSV column is consistently nullable.
    """
    by_key: dict[_KeyTuple, dict[int, int]] = {}
    for cr in confidence_records:
        key: _KeyTuple = (
            cr["question_id"],
            cr["condition"],
            cr["model_id"],
            int(cr.get("trial_id", 0)),
        )
        index_map = by_key.setdefault(key, {})
        for entry in cr.get("confidences", []) or []:
            idx = entry.get("citation_index")
            if isinstance(idx, int):
                index_map[idx] = entry.get("confidence")
    for row in verification_rows:
        key = (
            row["question_id"],
            row["condition"],
            row["model_id"],
            row["trial_id"],
        )
        idx = row["citation_index"]
        confidence = by_key.get(key, {}).get(idx)
        yield {**row, "confidence": confidence}


# ---- J2a judge join -----------------------------------------------------------
#
# The Path 7 judge fires only on verifier-FABRICATED / -UNVERIFIABLE citations
# (restricted-firing J2a). Verifier-REAL is accepted at face value. The judge
# output is keyed by the 5-part custom_id {qid}__{model}__{cond}__t{trial}__c{idx}.


def parse_judgment_custom_id(custom_id: str) -> _CitationKey:
    """Split a 5-part judge custom_id into its component fields.

    Fields are joined by `__`; the qid itself contains single underscores and
    the model id contains single hyphens, so a `__` split is unambiguous.
    The trial token is `t{n}` and the citation token is `c{n}`.
    """
    parts = custom_id.split("__")
    if len(parts) != 5:
        raise ValueError(f"expected 5 `__`-delimited parts, got {len(parts)}: {custom_id}")
    qid, model_id, condition, trial_tok, cite_tok = parts
    return qid, model_id, condition, int(trial_tok[1:]), int(cite_tok[1:])


def combine_j2a(verifier_status: str, judge_status: str | None) -> str:
    """Apply the J2a combine rule to one citation.

    verifier_status is the flattened (R-taxonomy) verifier label. judge_status
    is the raw judge verdict (REAL / FABRICATED / UNVERIFIABLE) or None when the
    judge did not fire or errored. Rule: verifier-REAL is accepted untouched;
    for verifier-FAB/-UNV the judge wins on REAL/FABRICATED and the verifier
    label stands when the judge says UNVERIFIABLE or errored.
    """
    if verifier_status == "REAL_CONFIRMED":
        return "REAL_CONFIRMED"
    if judge_status == "REAL":
        return "REAL_CONFIRMED"
    if judge_status == "FABRICATED":
        return "FABRICATED"
    return verifier_status  # judge UNVERIFIABLE / None -> verifier fallback


def build_judgment_index(
    judgments: Iterable[dict[str, Any]],
) -> dict[_CitationKey, str | None]:
    """Index judge verdicts by citation key. Errored rows map to None so the
    combine step falls back to the verifier label."""
    index: dict[_CitationKey, str | None] = {}
    for record in judgments:
        key = parse_judgment_custom_id(record["custom_id"])
        status = record.get("judge_status")
        if record.get("error"):
            status = None
        index[key] = status
    return index


def apply_j2a(
    flat_rows: Iterable[dict[str, Any]],
    judgment_index: dict[_CitationKey, str | None],
) -> Iterator[dict[str, Any]]:
    """Merge judge verdicts onto flattened verification rows.

    Adds `verifier_status` (the pre-judge label) and overwrites `status` with
    the J2a-combined label, so `status` is the primary Phase B outcome and
    `verifier_status` preserves the verifier-only audit column.
    """
    for row in flat_rows:
        verifier_status = row["status"]
        key: _CitationKey = (
            row["question_id"],
            row["model_id"],
            row["condition"],
            row["trial_id"],
            row["citation_index"],
        )
        judge_status = judgment_index.get(key)
        yield {
            **row,
            "verifier_status": verifier_status,
            "status": combine_j2a(verifier_status, judge_status),
        }


# ---- Descriptive fabrication-rate table ---------------------------------------


def _classify_generation(statuses: list[str]) -> int | None:
    """Per-generation outcome (mirrors R classify_fabricated).

    Any bad citation -> 1; >=1 good and no bad -> 0; cited only UNVERIFIABLE
    -> None (excluded). An empty status list (abstention) is handled by the
    caller as 0.
    """
    if any(s in _BAD_STATUSES for s in statuses):
        return 1
    if any(s in _GOOD_STATUSES for s in statuses):
        return 0
    return None  # cited only UNVERIFIABLE -> excluded


def fabrication_rate_table(
    verification_rows: Iterable[dict[str, Any]],
    questions: Iterable[dict[str, Any]],
    *,
    status_field: str,
    abstentions: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Per-(condition × cell × domain) generation-level fabrication rate.

    `status_field` selects the label column to score ("status" for the J2a
    primary, "verifier_status" for the verifier-only audit). Generations citing
    only UNVERIFIABLE sources are excluded; abstentions (zero-citation
    generations, passed explicitly since they leave no citation rows) count as
    honest non-fabrication (0) and stay in the denominator.
    """
    cell_domain = {q["id"]: (q["cell"], q["domain"]) for q in questions}

    # Collect each generation's citation statuses.
    by_gen: dict[_KeyTuple, list[str]] = {}
    for row in verification_rows:
        key: _KeyTuple = (
            row["question_id"], row["condition"], row["model_id"], row["trial_id"],
        )
        by_gen.setdefault(key, []).append(row[status_field])
    for ab in abstentions or []:
        key = (ab["question_id"], ab["condition"], ab["model_id"], int(ab["trial_id"]))
        by_gen.setdefault(key, [])  # empty -> abstention

    # Aggregate to (condition, cell, domain). The UNV-only exclusion depends on
    # the label set scored, so each block's denominator differs; n_excluded
    # makes that shift explicit per stratum instead of a silent NA drop.
    groups: dict[tuple[str, str, str], dict[str, int]] = {}
    for (qid, condition, _model, _trial), statuses in by_gen.items():
        cd = cell_domain.get(qid)
        if cd is None:
            continue
        cell, domain = cd
        g = groups.setdefault((condition, cell, domain),
                              {"n_generations": 0, "n_fabricated": 0, "n_excluded": 0})
        outcome = 0 if not statuses else _classify_generation(statuses)
        if outcome is None:
            g["n_excluded"] += 1  # cited-only-UNVERIFIABLE -> excluded
            continue
        g["n_generations"] += 1
        g["n_fabricated"] += outcome

    table: list[dict[str, Any]] = []
    for (condition, cell, domain), g in sorted(groups.items()):
        n = g["n_generations"]
        table.append({
            "condition": condition,
            "cell": cell,
            "domain": domain,
            "n_generations": n,
            "n_fabricated": g["n_fabricated"],
            "fabrication_rate": (g["n_fabricated"] / n) if n else 0.0,
            "n_excluded": g["n_excluded"],
        })
    return table


def judge_rescue_table(
    j2a_rows: Iterable[dict[str, Any]],
    judgment_index: dict[_CitationKey, str | None],
    questions: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Judge verdict distribution on judge-fired citations, per
    (verifier_status × condition × cell × domain).

    The restricted-firing judge sees only verifier-FABRICATED / -UNVERIFIABLE
    citations; this table shows where its REAL "rescues" land, so an aggregate
    rescue rate can be checked against the verifier's known per-stratum
    precision instead of being read as one opaque number.
    """
    cell_domain = {q["id"]: (q["cell"], q["domain"]) for q in questions}
    groups: dict[tuple[str, str, str, str], dict[str, int]] = {}
    for row in j2a_rows:
        verifier_status = row["verifier_status"]
        if verifier_status not in ("FABRICATED", "UNVERIFIABLE"):
            continue
        cd = cell_domain.get(row["question_id"])
        if cd is None:
            continue
        cell, domain = cd
        key: _CitationKey = (
            row["question_id"], row["model_id"], row["condition"],
            row["trial_id"], row["citation_index"],
        )
        verdict = judgment_index.get(key)
        g = groups.setdefault(
            (verifier_status, row["condition"], cell, domain),
            {"n_fired": 0, "n_judge_real": 0, "n_judge_fabricated": 0,
             "n_judge_unverifiable": 0, "n_judge_missing": 0},
        )
        g["n_fired"] += 1
        if verdict == "REAL":
            g["n_judge_real"] += 1
        elif verdict == "FABRICATED":
            g["n_judge_fabricated"] += 1
        elif verdict == "UNVERIFIABLE":
            g["n_judge_unverifiable"] += 1
        else:
            g["n_judge_missing"] += 1

    table: list[dict[str, Any]] = []
    for (verifier_status, condition, cell, domain), g in sorted(groups.items()):
        table.append({
            "verifier_status": verifier_status,
            "condition": condition,
            "cell": cell,
            "domain": domain,
            **g,
            "rescue_rate": g["n_judge_real"] / g["n_fired"],
        })
    return table


def write_csv(
    rows: Iterable[dict[str, Any]],
    path: Path,
    *,
    columns: list[str],
) -> None:
    """Write `rows` as a DictWriter CSV with the given column order.

    None values are written as empty cells (read back as NA by R's
    `read.csv(..., na.strings = "")` default-compatible code path).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def _abstention_generations(
    generations: Iterable[dict[str, Any]],
    citation_rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generations that produced zero citations (honest abstentions).

    These leave no rows in the flattened verification stream, so the rate
    table needs them passed explicitly to land in the denominator as 0.
    """
    cited = {
        (r["question_id"], r["condition"], r["model_id"], r["trial_id"])
        for r in citation_rows
    }
    out: list[dict[str, Any]] = []
    for g in generations:
        key = (g["question_id"], g["condition"], g["model_id"], g["trial_id"])
        if key not in cited:
            out.append({
                "question_id": g["question_id"],
                "condition": g["condition"],
                "model_id": g["model_id"],
                "trial_id": g["trial_id"],
            })
    return out


def run(
    *,
    questions_path: Path,
    registry_path: Path,
    generations_paths: list[Path],
    verifications_paths: list[Path],
    confidences_paths: list[Path],
    output_dir: Path,
    judgments_paths: list[Path] | None = None,
) -> None:
    """Join all artifacts and write the R-input CSVs + the descriptive table.

    When `judgments_paths` is given, the J2a judge verdicts are merged onto the
    verification rows: `status` becomes the J2a-combined label and a
    `verifier_status` column preserves the verifier-only audit label. With no
    judgments the two columns are identical (backward compatible).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # questions.csv + models.csv (small reference frames).
    write_csv(
        to_questions_frame(questions_path),
        output_dir / "questions.csv",
        columns=["id", "cell", "domain"],
    )
    write_csv(
        to_models_frame(registry_path),
        output_dir / "models.csv",
        columns=["model_id", "family", "is_post_trained", "is_open_weights"],
    )

    # generations.csv (concatenate per-model files).
    gens: list[dict[str, Any]] = []
    for p in generations_paths:
        gens.extend(to_generations_frame(p))
    write_csv(
        gens,
        output_dir / "generations.csv",
        columns=[
            "question_id",
            "condition",
            "model_id",
            "trial_id",
            "output_text",
        ],
    )

    # verifications.csv: flatten -> J2a merge -> attach confidence -> write.
    verif_records: list[dict[str, Any]] = []
    for p in verifications_paths:
        verif_records.extend(_read_jsonl(p))
    judgment_records: list[dict[str, Any]] = []
    for p in judgments_paths or []:
        judgment_records.extend(_read_jsonl(p))
    confidence_records: list[dict[str, Any]] = []
    for p in confidences_paths:
        confidence_records.extend(_read_jsonl(p))

    judgment_index = build_judgment_index(judgment_records)
    flat = list(flatten_verifications(verif_records))
    j2a_rows = list(apply_j2a(flat, judgment_index))
    joined = list(attach_confidences(j2a_rows, confidence_records))
    write_csv(
        joined,
        output_dir / "verifications.csv",
        columns=[
            "question_id",
            "condition",
            "model_id",
            "trial_id",
            "citation_index",
            "status",
            "verifier_status",
            "confidence",
        ],
    )

    # fabrication_rates.csv: the descriptive Phase B headline, one block per
    # label (verifier_only / j2a). The EXISTENT_WRONG_ID-corrected column is
    # populated separately once the sub-study (amendment #6) runs.
    questions = to_questions_frame(questions_path)
    abstentions = _abstention_generations(gens, j2a_rows)
    rate_rows: list[dict[str, Any]] = []
    for label, field in (("verifier_only", "verifier_status"), ("j2a", "status")):
        for entry in fabrication_rate_table(
            j2a_rows, questions, status_field=field, abstentions=abstentions
        ):
            rate_rows.append({"label": label, **entry})
    write_csv(
        rate_rows,
        output_dir / "fabrication_rates.csv",
        columns=[
            "label",
            "condition",
            "cell",
            "domain",
            "n_generations",
            "n_fabricated",
            "fabrication_rate",
            "n_excluded",
        ],
    )

    # judge_rescue_rates.csv: where the restricted-firing judge's REAL rescues
    # land, per verifier label that triggered firing (methods supplement).
    if judgment_records:
        write_csv(
            judge_rescue_table(j2a_rows, judgment_index, questions),
            output_dir / "judge_rescue_rates.csv",
            columns=[
                "verifier_status",
                "condition",
                "cell",
                "domain",
                "n_fired",
                "n_judge_real",
                "n_judge_fabricated",
                "n_judge_unverifiable",
                "n_judge_missing",
                "rescue_rate",
            ],
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.analysis.aggregate",
        description="Phase B JSONL -> R-input CSV aggregation.",
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--generations", type=Path, required=True, nargs="+",
        help="One or more generations.jsonl files (one per model).",
    )
    parser.add_argument(
        "--verifications", type=Path, required=True, nargs="+",
        help="One or more verifications.jsonl files (one per model).",
    )
    parser.add_argument(
        "--confidences", type=Path, required=True, nargs="+",
        help="One or more confidences.jsonl files (one per model).",
    )
    parser.add_argument(
        "--judgments", type=Path, default=[], nargs="+",
        help="One or more J2a judgment JSONL files (restricted-firing judge "
             "verdicts keyed by 5-part custom_id). Optional.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    run(
        questions_path=args.questions,
        registry_path=args.registry,
        generations_paths=args.generations,
        verifications_paths=args.verifications,
        confidences_paths=args.confidences,
        judgments_paths=args.judgments,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
