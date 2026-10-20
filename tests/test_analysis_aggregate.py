"""Contract tests for src/analysis/aggregate.py (Phase B → R join layer).

Produces the four frames `build_e2_data()` in
src/analysis/e2_prompt_manipulation.R reads:
  - generations.csv   (one row per generation; question_id, condition,
                       model_id, trial_id, output_text)
  - verifications.csv (one row per CITATION — the R code aggregates these
                       by (qid, condition, model_id, trial_id) status)
  - questions.csv     (one row per question; id, cell, domain)
  - models.csv        (one row per model; model_id, family,
                       is_post_trained, is_open_weights)

Status mapping: Python pipeline uses {REAL, FABRICATED, UNVERIFIABLE};
the R aggregator's GOOD_STATUSES treats REAL_CONFIRMED as the success
state. We map REAL -> REAL_CONFIRMED at flatten time so the existing
R code consumes Phase B output without changes. MISGROUNDED is Phase H
territory and does not appear in Phase B output.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


# ---------- flatten_verifications ----------


@pytest.fixture
def two_verifications() -> list[dict]:
    return [
        {
            "question_id": "academic_known_real_0000",
            "condition": "citations_expected",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "n_citations": 2,
            "citation_statuses": ["REAL", "FABRICATED"],
            "verified_at": "2026-06-04T05:30:00+00:00",
        },
        {
            "question_id": "legal_unanswerable_0001",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 3,
            "n_citations": 0,
            "citation_statuses": [],
            "verified_at": "2026-06-04T05:30:00+00:00",
        },
    ]


def test_flatten_verifications_explodes_per_citation_row(two_verifications):
    from src.analysis.aggregate import flatten_verifications

    rows = list(flatten_verifications(two_verifications))
    # 2 cites from generation 1; 0 from generation 2.
    assert len(rows) == 2
    assert rows[0]["question_id"] == "academic_known_real_0000"
    assert rows[0]["trial_id"] == 0
    assert rows[0]["citation_index"] == 0
    assert rows[1]["citation_index"] == 1


def test_flatten_verifications_maps_real_to_real_confirmed(two_verifications):
    """The R aggregator's GOOD_STATUSES expects REAL_CONFIRMED, not REAL."""
    from src.analysis.aggregate import flatten_verifications

    rows = list(flatten_verifications(two_verifications))
    statuses = [r["status"] for r in rows]
    assert "REAL_CONFIRMED" in statuses
    assert "REAL" not in statuses


def test_flatten_verifications_preserves_fabricated_and_unverifiable():
    from src.analysis.aggregate import flatten_verifications

    verif = [
        {
            "question_id": "q-1",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "n_citations": 3,
            "citation_statuses": ["FABRICATED", "UNVERIFIABLE", "REAL"],
            "verified_at": "x",
        }
    ]
    rows = list(flatten_verifications(verif))
    assert [r["status"] for r in rows] == [
        "FABRICATED",
        "UNVERIFIABLE",
        "REAL_CONFIRMED",
    ]


def test_flatten_verifications_zero_citation_records_emit_nothing():
    from src.analysis.aggregate import flatten_verifications

    verif = [
        {
            "question_id": "q-1",
            "condition": "abstention_permitted",
            "model_id": "m",
            "trial_id": 0,
            "n_citations": 0,
            "citation_statuses": [],
            "verified_at": "x",
        }
    ]
    rows = list(flatten_verifications(verif))
    assert rows == []


# ---------- to_questions_frame ----------


def test_to_questions_frame_extracts_id_cell_domain(tmp_path: Path):
    from src.analysis.aggregate import to_questions_frame

    records = [
        {
            "id": "academic_known_real_0000",
            "domain": "academic",
            "question": "What is X?",
            "expected_existence": "answerable",
            "cell": "known_real",
            "expected_citation": {"title": "X"},
        },
        {
            "id": "legal_unanswerable_0001",
            "domain": "legal",
            "question": "Cite a fictional case.",
            "expected_existence": "unanswerable",
            "cell": "unanswerable",
        },
    ]
    path = tmp_path / "q.jsonl"
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    frame = to_questions_frame(path)
    assert len(frame) == 2
    assert frame[0] == {
        "id": "academic_known_real_0000",
        "cell": "known_real",
        "domain": "academic",
    }


# ---------- to_models_frame ----------


def test_to_models_frame_passes_through_registry(tmp_path: Path):
    from src.analysis.aggregate import to_models_frame

    records = [
        {
            "model_id": "claude-haiku-4-5-20251001",
            "family": "anthropic",
            "is_post_trained": 1,
            "is_open_weights": 0,
            "phase_b_role": "step1_haiku_full_900q",
            "pin_kind": "dated_build",
        }
    ]
    path = tmp_path / "registry.jsonl"
    path.write_text(json.dumps(records[0]) + "\n", encoding="utf-8")
    frame = to_models_frame(path)
    assert frame == [
        {
            "model_id": "claude-haiku-4-5-20251001",
            "family": "anthropic",
            "is_post_trained": 1,
            "is_open_weights": 0,
        }
    ]


# ---------- to_generations_frame ----------


def test_to_generations_frame_passes_through_with_output_text(tmp_path: Path):
    from src.analysis.aggregate import to_generations_frame

    gens = [
        {
            "question_id": "academic_known_real_0000",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "output_text": "Vaswani et al. 2017.",
            "input_tokens": 30,
            "output_tokens": 20,
            "finish_reason": "end_turn",
            "timestamp": "2026-06-04T04:00:00+00:00",
        }
    ]
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(gens[0]) + "\n", encoding="utf-8")
    frame = to_generations_frame(path)
    assert frame == [
        {
            "question_id": "academic_known_real_0000",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "output_text": "Vaswani et al. 2017.",
        }
    ]


def test_to_generations_frame_skips_error_rows(tmp_path: Path):
    """Generation rows with `error` carry no analyzable output_text."""
    from src.analysis.aggregate import to_generations_frame

    gens = [
        {
            "question_id": "q-1",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "output_text": "",
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": None,
            "timestamp": "x",
            "error": "expired",
        }
    ]
    path = tmp_path / "g.jsonl"
    path.write_text(json.dumps(gens[0]) + "\n", encoding="utf-8")
    assert to_generations_frame(path) == []


# ---------- attach_confidences ----------


def test_attach_confidences_joins_by_key_and_index():
    from src.analysis.aggregate import attach_confidences

    verif_rows = [
        {
            "question_id": "q-1",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "citation_index": 0,
            "status": "REAL_CONFIRMED",
        },
        {
            "question_id": "q-1",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "citation_index": 1,
            "status": "FABRICATED",
        },
    ]
    confidence_records = [
        {
            "question_id": "q-1",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "confidences": [
                {"citation_index": 0, "confidence": 95},
                {"citation_index": 1, "confidence": 30},
            ],
            "queried_at": "x",
        }
    ]
    out = list(attach_confidences(verif_rows, confidence_records))
    assert out[0]["confidence"] == 95
    assert out[1]["confidence"] == 30


def test_attach_confidences_missing_record_yields_none():
    from src.analysis.aggregate import attach_confidences

    verif_rows = [
        {
            "question_id": "q-1",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "citation_index": 0,
            "status": "FABRICATED",
        }
    ]
    # No confidence record for this generation (e.g., zero-citation skip).
    out = list(attach_confidences(verif_rows, []))
    assert out[0]["confidence"] is None


def test_attach_confidences_missing_index_within_record_yields_none():
    from src.analysis.aggregate import attach_confidences

    verif_rows = [
        {
            "question_id": "q-1",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "citation_index": 1,
            "status": "FABRICATED",
        }
    ]
    confidence_records = [
        {
            "question_id": "q-1",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "confidences": [{"citation_index": 0, "confidence": 95}],  # only 0
            "queried_at": "x",
        }
    ]
    out = list(attach_confidences(verif_rows, confidence_records))
    assert out[0]["confidence"] is None


# ---------- write_csv ----------


def test_write_csv_writes_header_and_rows(tmp_path: Path):
    from src.analysis.aggregate import write_csv

    rows = [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "y"},
    ]
    path = tmp_path / "out.csv"
    write_csv(rows, path, columns=["a", "b"])
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        out = list(reader)
    assert out[0] == {"a": "1", "b": "x"}
    assert out[1] == {"a": "2", "b": "y"}


def test_write_csv_handles_empty(tmp_path: Path):
    from src.analysis.aggregate import write_csv

    path = tmp_path / "out.csv"
    write_csv([], path, columns=["a", "b"])
    text = path.read_text(encoding="utf-8")
    # Header line only.
    assert text.strip() == "a,b"


# ---------- End-to-end run() ----------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_run_writes_four_csvs_with_expected_columns(tmp_path: Path):
    from src.analysis.aggregate import run

    questions = [
        {
            "id": "academic_known_real_0000",
            "domain": "academic",
            "question": "What is X?",
            "expected_existence": "answerable",
            "cell": "known_real",
        }
    ]
    registry = [
        {
            "model_id": "claude-haiku-4-5-20251001",
            "family": "anthropic",
            "is_post_trained": 1,
            "is_open_weights": 0,
            "phase_b_role": "step1_haiku_full_900q",
            "pin_kind": "dated_build",
        }
    ]
    generations = [
        {
            "question_id": "academic_known_real_0000",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "output_text": "Vaswani et al. 2017.",
            "input_tokens": 30,
            "output_tokens": 20,
            "finish_reason": "end_turn",
            "timestamp": "2026-06-04T04:00:00+00:00",
        }
    ]
    verifications = [
        {
            "question_id": "academic_known_real_0000",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "n_citations": 1,
            "citation_statuses": ["REAL"],
            "verified_at": "2026-06-04T05:30:00+00:00",
        }
    ]
    confidences = [
        {
            "question_id": "academic_known_real_0000",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "confidences": [{"citation_index": 0, "confidence": 88}],
            "queried_at": "2026-06-04T06:00:00+00:00",
        }
    ]
    _write_jsonl(tmp_path / "q.jsonl", questions)
    _write_jsonl(tmp_path / "registry.jsonl", registry)
    _write_jsonl(tmp_path / "g.jsonl", generations)
    _write_jsonl(tmp_path / "v.jsonl", verifications)
    _write_jsonl(tmp_path / "c.jsonl", confidences)
    out_dir = tmp_path / "out"

    run(
        questions_path=tmp_path / "q.jsonl",
        registry_path=tmp_path / "registry.jsonl",
        generations_paths=[tmp_path / "g.jsonl"],
        verifications_paths=[tmp_path / "v.jsonl"],
        confidences_paths=[tmp_path / "c.jsonl"],
        output_dir=out_dir,
    )
    assert (out_dir / "generations.csv").exists()
    assert (out_dir / "verifications.csv").exists()
    assert (out_dir / "questions.csv").exists()
    assert (out_dir / "models.csv").exists()

    # Verifications CSV has one row per citation, with citation_index +
    # confidence joined and REAL mapped to REAL_CONFIRMED.
    with (out_dir / "verifications.csv").open(newline="", encoding="utf-8") as fh:
        verif_rows = list(csv.DictReader(fh))
    assert len(verif_rows) == 1
    assert verif_rows[0]["status"] == "REAL_CONFIRMED"
    assert verif_rows[0]["confidence"] == "88"

    # Questions CSV: id / cell / domain only (build_e2_data merge contract).
    with (out_dir / "questions.csv").open(newline="", encoding="utf-8") as fh:
        q_rows = list(csv.DictReader(fh))
    assert set(q_rows[0].keys()) == {"id", "cell", "domain"}


# ---------- J2a judge join (added with the Path 7 judge layer) ----------
#
# The judge fires only on verifier-FABRICATED / -UNVERIFIABLE citations
# (restricted-firing J2a). Verifier-REAL is accepted at face value (the judge
# never ran for it). custom_id is the 5-part judge key
# {qid}__{model}__{cond}__t{trial}__c{cite_idx}.


def test_parse_judgment_custom_id_splits_five_parts():
    from src.analysis.aggregate import parse_judgment_custom_id

    cid = "legal_unanswerable_0545__claude-haiku-4-5-20251001__citations_expected__t0__c3"
    key = parse_judgment_custom_id(cid)
    assert key == (
        "legal_unanswerable_0545",
        "claude-haiku-4-5-20251001",
        "citations_expected",
        0,
        3,
    )


def test_parse_judgment_custom_id_keeps_qid_underscores():
    """qid has internal single underscores; the field delimiter is `__`."""
    from src.analysis.aggregate import parse_judgment_custom_id

    cid = "medical_known_real_0657__claude-haiku-4-5-20251001__length_matched__t4__c0"
    qid, model, cond, trial, cidx = parse_judgment_custom_id(cid)
    assert qid == "medical_known_real_0657"
    assert cond == "length_matched"
    assert trial == 4
    assert cidx == 0


@pytest.mark.parametrize(
    "verifier_status, judge_status, expected",
    [
        # Verifier-REAL: judge never fires; accepted as-is.
        ("REAL_CONFIRMED", None, "REAL_CONFIRMED"),
        # Verifier-FABRICATED + judge verdicts.
        ("FABRICATED", "REAL", "REAL_CONFIRMED"),
        ("FABRICATED", "FABRICATED", "FABRICATED"),
        ("FABRICATED", "UNVERIFIABLE", "FABRICATED"),  # verifier fallback
        ("FABRICATED", None, "FABRICATED"),  # judge errored -> fallback
        # Verifier-UNVERIFIABLE + judge verdicts.
        ("UNVERIFIABLE", "REAL", "REAL_CONFIRMED"),
        ("UNVERIFIABLE", "FABRICATED", "FABRICATED"),
        ("UNVERIFIABLE", "UNVERIFIABLE", "UNVERIFIABLE"),  # verifier fallback
        ("UNVERIFIABLE", None, "UNVERIFIABLE"),  # judge errored -> fallback
    ],
)
def test_combine_j2a_truth_table(verifier_status, judge_status, expected):
    from src.analysis.aggregate import combine_j2a

    assert combine_j2a(verifier_status, judge_status) == expected


def test_combine_j2a_ignores_stray_judge_on_verifier_real():
    """A verifier-REAL citation must stay REAL_CONFIRMED even if a judgment
    row exists for it (defensive: J2a never sends REAL to the judge)."""
    from src.analysis.aggregate import combine_j2a

    assert combine_j2a("REAL_CONFIRMED", "FABRICATED") == "REAL_CONFIRMED"


def test_build_judgment_index_keys_on_full_tuple():
    from src.analysis.aggregate import build_judgment_index

    judgments = [
        {
            "custom_id": "q_known_real_0001__m__neutral__t0__c0",
            "judge_status": "FABRICATED",
            "error": None,
        },
        {
            "custom_id": "q_known_real_0001__m__neutral__t0__c1",
            "judge_status": "REAL",
            "error": None,
        },
    ]
    index = build_judgment_index(judgments)
    assert index[("q_known_real_0001", "m", "neutral", 0, 0)] == "FABRICATED"
    assert index[("q_known_real_0001", "m", "neutral", 0, 1)] == "REAL"


def test_build_judgment_index_errored_rows_map_to_none():
    """A judge row with an error (or missing status) yields None so the
    combine step falls back to the verifier label."""
    from src.analysis.aggregate import build_judgment_index

    judgments = [
        {
            "custom_id": "q__m__neutral__t0__c0",
            "judge_status": None,
            "error": "null_verdict",
        }
    ]
    index = build_judgment_index(judgments)
    assert index[("q", "m", "neutral", 0, 0)] is None


def test_apply_j2a_sets_status_and_preserves_verifier_status():
    from src.analysis.aggregate import apply_j2a

    flat_rows = [
        {
            "question_id": "q",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "citation_index": 0,
            "status": "FABRICATED",  # verifier said FAB
        },
        {
            "question_id": "q",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "citation_index": 1,
            "status": "REAL_CONFIRMED",  # verifier said REAL; judge never fired
        },
    ]
    judgment_index = {("q", "m", "neutral", 0, 0): "REAL"}  # judge rescued cite 0
    out = list(apply_j2a(flat_rows, judgment_index))
    # cite 0: verifier FAB, judge REAL -> J2a REAL_CONFIRMED, verifier preserved
    assert out[0]["status"] == "REAL_CONFIRMED"
    assert out[0]["verifier_status"] == "FABRICATED"
    # cite 1: verifier REAL, no judgment -> unchanged
    assert out[1]["status"] == "REAL_CONFIRMED"
    assert out[1]["verifier_status"] == "REAL_CONFIRMED"


def test_apply_j2a_fab_with_no_judgment_falls_back_to_verifier():
    from src.analysis.aggregate import apply_j2a

    flat_rows = [
        {
            "question_id": "q",
            "condition": "neutral",
            "model_id": "m",
            "trial_id": 0,
            "citation_index": 0,
            "status": "FABRICATED",
        }
    ]
    out = list(apply_j2a(flat_rows, {}))  # no judgment for this cite
    assert out[0]["status"] == "FABRICATED"
    assert out[0]["verifier_status"] == "FABRICATED"


# ---------- fabrication_rate_table (descriptive Phase B headline) ----------


def _verif_row(qid, cond, trial, cidx, status, verifier_status=None):
    return {
        "question_id": qid,
        "condition": cond,
        "model_id": "m",
        "trial_id": trial,
        "citation_index": cidx,
        "status": status,
        "verifier_status": verifier_status if verifier_status is not None else status,
    }


def test_fabrication_rate_table_generation_level_any_bad():
    """Fabrication is a per-GENERATION outcome: a generation counts as
    fabricated if ANY of its citations is FABRICATED (R classify_fabricated)."""
    from src.analysis.aggregate import fabrication_rate_table

    questions = [
        {"id": "academic_obscure_real_0001", "cell": "obscure_real", "domain": "academic"},
        {"id": "academic_obscure_real_0002", "cell": "obscure_real", "domain": "academic"},
    ]
    # gen 1 (q1,neutral,t0): one FAB + one REAL -> fabricated=1
    # gen 2 (q2,neutral,t0): one REAL only          -> fabricated=0
    rows = [
        _verif_row("academic_obscure_real_0001", "neutral", 0, 0, "FABRICATED"),
        _verif_row("academic_obscure_real_0001", "neutral", 0, 1, "REAL_CONFIRMED"),
        _verif_row("academic_obscure_real_0002", "neutral", 0, 0, "REAL_CONFIRMED"),
    ]
    table = fabrication_rate_table(rows, questions, status_field="status")
    cell = next(
        r for r in table
        if r["condition"] == "neutral" and r["cell"] == "obscure_real"
        and r["domain"] == "academic"
    )
    assert cell["n_generations"] == 2
    assert cell["n_fabricated"] == 1
    assert cell["fabrication_rate"] == pytest.approx(0.5)


def test_fabrication_rate_table_verifier_vs_j2a_differ():
    """Same generations, scored under verifier_status vs J2a status, give
    different rates — the whole point of the three-column audit."""
    from src.analysis.aggregate import fabrication_rate_table

    questions = [
        {"id": "academic_obscure_real_0001", "cell": "obscure_real", "domain": "academic"},
    ]
    # Verifier called it FAB; J2a judge rescued it to REAL.
    rows = [
        _verif_row(
            "academic_obscure_real_0001", "neutral", 0, 0,
            status="REAL_CONFIRMED", verifier_status="FABRICATED",
        ),
    ]
    verifier_tbl = fabrication_rate_table(rows, questions, status_field="verifier_status")
    j2a_tbl = fabrication_rate_table(rows, questions, status_field="status")
    assert verifier_tbl[0]["fabrication_rate"] == pytest.approx(1.0)
    assert j2a_tbl[0]["fabrication_rate"] == pytest.approx(0.0)


def test_fabrication_rate_table_cited_only_unverifiable_excluded():
    """A generation citing only UNVERIFIABLE sources is NA/excluded (R rule),
    so it contributes to neither numerator nor denominator. The exclusion is
    COUNTED per stratum (n_excluded) so the verifier_only and j2a blocks can't
    be misread as same-denominator."""
    from src.analysis.aggregate import fabrication_rate_table

    questions = [
        {"id": "medical_unanswerable_0001", "cell": "unanswerable", "domain": "medical"},
        {"id": "medical_unanswerable_0002", "cell": "unanswerable", "domain": "medical"},
    ]
    rows = [
        _verif_row("medical_unanswerable_0001", "neutral", 0, 0, "UNVERIFIABLE"),
        _verif_row("medical_unanswerable_0002", "neutral", 0, 0, "FABRICATED"),
    ]
    table = fabrication_rate_table(rows, questions, status_field="status")
    cell = table[0]
    # Only the FABRICATED generation counts; the UNVERIFIABLE-only one excluded.
    assert cell["n_generations"] == 1
    assert cell["n_fabricated"] == 1
    assert cell["n_excluded"] == 1


def test_fabrication_rate_table_n_excluded_zero_when_nothing_excluded():
    from src.analysis.aggregate import fabrication_rate_table

    questions = [
        {"id": "academic_obscure_real_0001", "cell": "obscure_real", "domain": "academic"},
    ]
    rows = [_verif_row("academic_obscure_real_0001", "neutral", 0, 0, "REAL_CONFIRMED")]
    table = fabrication_rate_table(rows, questions, status_field="status")
    assert table[0]["n_excluded"] == 0


# ---------- judge_rescue_table ----------
#
# Supplement table: judge verdict distribution on the citations it fired on,
# split by the verifier label that triggered firing. Checks whether the
# aggregate judge-REAL "rescue" rate is concentrated where the verifier's known
# low FAB precision predicts, or is anomalously uniform.


def _q(qid, cell, domain):
    return {"id": qid, "cell": cell, "domain": domain}


def test_judge_rescue_table_groups_by_verifier_status_and_stratum():
    from src.analysis.aggregate import judge_rescue_table

    questions = [_q("academic_obscure_real_0001", "obscure_real", "academic")]
    rows = [
        _verif_row("academic_obscure_real_0001", "neutral", 0, 0,
                   status="REAL_CONFIRMED", verifier_status="FABRICATED"),
        _verif_row("academic_obscure_real_0001", "neutral", 0, 1,
                   status="FABRICATED", verifier_status="FABRICATED"),
        _verif_row("academic_obscure_real_0001", "neutral", 0, 2,
                   status="REAL_CONFIRMED", verifier_status="UNVERIFIABLE"),
    ]
    index = {
        ("academic_obscure_real_0001", "m", "neutral", 0, 0): "REAL",
        ("academic_obscure_real_0001", "m", "neutral", 0, 1): "FABRICATED",
        ("academic_obscure_real_0001", "m", "neutral", 0, 2): "REAL",
    }
    table = judge_rescue_table(rows, index, questions)
    by_vs = {r["verifier_status"]: r for r in table}
    fab = by_vs["FABRICATED"]
    assert (fab["condition"], fab["cell"], fab["domain"]) == (
        "neutral", "obscure_real", "academic")
    assert fab["n_fired"] == 2
    assert fab["n_judge_real"] == 1
    assert fab["n_judge_fabricated"] == 1
    assert fab["rescue_rate"] == pytest.approx(0.5)
    unv = by_vs["UNVERIFIABLE"]
    assert unv["n_fired"] == 1
    assert unv["rescue_rate"] == pytest.approx(1.0)


def test_judge_rescue_table_ignores_verifier_real_rows():
    from src.analysis.aggregate import judge_rescue_table

    questions = [_q("academic_known_real_0000", "known_real", "academic")]
    rows = [
        _verif_row("academic_known_real_0000", "neutral", 0, 0, "REAL_CONFIRMED"),
    ]
    assert judge_rescue_table(rows, {}, questions) == []


def test_judge_rescue_table_counts_missing_judgments():
    """A judge-eligible citation with no verdict (errored or never returned)
    lands in n_judge_missing, not in any verdict bucket."""
    from src.analysis.aggregate import judge_rescue_table

    questions = [_q("medical_unanswerable_0001", "unanswerable", "medical")]
    rows = [
        _verif_row("medical_unanswerable_0001", "neutral", 0, 0,
                   status="UNVERIFIABLE", verifier_status="UNVERIFIABLE"),
    ]
    table = judge_rescue_table(rows, {}, questions)
    assert table[0]["n_fired"] == 1
    assert table[0]["n_judge_missing"] == 1
    assert table[0]["rescue_rate"] == pytest.approx(0.0)


def test_fabrication_rate_table_counts_abstention_as_non_fabricated():
    """A generation with zero citations is honest non-fabrication (0), and
    must appear in the denominator. Abstentions are passed explicitly since
    they leave no citation rows."""
    from src.analysis.aggregate import fabrication_rate_table

    questions = [
        {"id": "academic_obscure_real_0001", "cell": "obscure_real", "domain": "academic"},
    ]
    # No citation rows for this generation; declared via abstentions list.
    abstentions = [
        {"question_id": "academic_obscure_real_0001", "condition": "neutral",
         "model_id": "m", "trial_id": 0},
    ]
    table = fabrication_rate_table([], questions, status_field="status",
                                   abstentions=abstentions)
    cell = table[0]
    assert cell["n_generations"] == 1
    assert cell["n_fabricated"] == 0
    assert cell["fabrication_rate"] == pytest.approx(0.0)


# ---------- run() with the optional judgments stream ----------


def test_run_with_judgments_writes_j2a_and_verifier_columns(tmp_path: Path):
    from src.analysis.aggregate import run

    questions = [
        {"id": "academic_obscure_real_0001", "domain": "academic",
         "question": "Q", "expected_existence": "answerable", "cell": "obscure_real"},
    ]
    registry = [
        {"model_id": "claude-haiku-4-5-20251001", "family": "anthropic",
         "is_post_trained": 1, "is_open_weights": 0},
    ]
    generations = [
        {"question_id": "academic_obscure_real_0001", "condition": "neutral",
         "model_id": "claude-haiku-4-5-20251001", "trial_id": 0,
         "output_text": "Some cite.", "timestamp": "x"},
    ]
    verifications = [
        {"question_id": "academic_obscure_real_0001", "condition": "neutral",
         "model_id": "claude-haiku-4-5-20251001", "trial_id": 0,
         "n_citations": 1, "citation_statuses": ["FABRICATED"], "verified_at": "x"},
    ]
    # The judge rescued the lone FAB citation to REAL.
    judgments = [
        {"custom_id": "academic_obscure_real_0001__claude-haiku-4-5-20251001__neutral__t0__c0",
         "judge_status": "REAL", "judge_reasoning": "found it",
         "judge_evidence_url": "https://x", "judge_search_count": 1, "error": None},
    ]
    confidences = []
    _write_jsonl(tmp_path / "q.jsonl", questions)
    _write_jsonl(tmp_path / "registry.jsonl", registry)
    _write_jsonl(tmp_path / "g.jsonl", generations)
    _write_jsonl(tmp_path / "v.jsonl", verifications)
    _write_jsonl(tmp_path / "j.jsonl", judgments)
    (tmp_path / "c.jsonl").write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"

    run(
        questions_path=tmp_path / "q.jsonl",
        registry_path=tmp_path / "registry.jsonl",
        generations_paths=[tmp_path / "g.jsonl"],
        verifications_paths=[tmp_path / "v.jsonl"],
        confidences_paths=[tmp_path / "c.jsonl"],
        judgments_paths=[tmp_path / "j.jsonl"],
        output_dir=out_dir,
    )
    with (out_dir / "verifications.csv").open(newline="", encoding="utf-8") as fh:
        verif_rows = list(csv.DictReader(fh))
    assert len(verif_rows) == 1
    # Primary `status` reflects the J2a rescue; verifier_status preserves the raw label.
    assert verif_rows[0]["status"] == "REAL_CONFIRMED"
    assert verif_rows[0]["verifier_status"] == "FABRICATED"

    # The descriptive Phase B headline: per-(condition×cell×domain) rates under
    # both label definitions, written as its own CSV.
    assert (out_dir / "fabrication_rates.csv").exists()
    with (out_dir / "fabrication_rates.csv").open(newline="", encoding="utf-8") as fh:
        rate_rows = list(csv.DictReader(fh))
    by_label = {r["label"]: r for r in rate_rows}
    assert by_label["verifier_only"]["fabrication_rate"] == "1.0"
    assert by_label["j2a"]["fabrication_rate"] == "0.0"
    # Denominator transparency: each block reports its own exclusion count.
    assert by_label["verifier_only"]["n_excluded"] == "0"
    assert by_label["j2a"]["n_excluded"] == "0"

    # Rescue table: the lone verifier-FAB citation was judge-rescued to REAL.
    with (out_dir / "judge_rescue_rates.csv").open(newline="", encoding="utf-8") as fh:
        rescue_rows = list(csv.DictReader(fh))
    assert len(rescue_rows) == 1
    assert rescue_rows[0]["verifier_status"] == "FABRICATED"
    assert rescue_rows[0]["n_fired"] == "1"
    assert rescue_rows[0]["n_judge_real"] == "1"
    assert rescue_rows[0]["rescue_rate"] == "1.0"


def test_run_without_judgments_leaves_verifier_status_equal_to_status(tmp_path: Path):
    """Backward compatible: with no judgments stream, J2a status == verifier."""
    from src.analysis.aggregate import run

    questions = [
        {"id": "academic_known_real_0000", "domain": "academic",
         "question": "Q", "expected_existence": "answerable", "cell": "known_real"},
    ]
    registry = [
        {"model_id": "claude-haiku-4-5-20251001", "family": "anthropic",
         "is_post_trained": 1, "is_open_weights": 0},
    ]
    generations = [
        {"question_id": "academic_known_real_0000", "condition": "neutral",
         "model_id": "claude-haiku-4-5-20251001", "trial_id": 0,
         "output_text": "Vaswani et al. 2017.", "timestamp": "x"},
    ]
    verifications = [
        {"question_id": "academic_known_real_0000", "condition": "neutral",
         "model_id": "claude-haiku-4-5-20251001", "trial_id": 0,
         "n_citations": 1, "citation_statuses": ["REAL"], "verified_at": "x"},
    ]
    _write_jsonl(tmp_path / "q.jsonl", questions)
    _write_jsonl(tmp_path / "registry.jsonl", registry)
    _write_jsonl(tmp_path / "g.jsonl", generations)
    _write_jsonl(tmp_path / "v.jsonl", verifications)
    (tmp_path / "c.jsonl").write_text("", encoding="utf-8")
    out_dir = tmp_path / "out"

    run(
        questions_path=tmp_path / "q.jsonl",
        registry_path=tmp_path / "registry.jsonl",
        generations_paths=[tmp_path / "g.jsonl"],
        verifications_paths=[tmp_path / "v.jsonl"],
        confidences_paths=[tmp_path / "c.jsonl"],
        output_dir=out_dir,
    )
    with (out_dir / "verifications.csv").open(newline="", encoding="utf-8") as fh:
        verif_rows = list(csv.DictReader(fh))
    assert verif_rows[0]["status"] == "REAL_CONFIRMED"
    assert verif_rows[0]["verifier_status"] == "REAL_CONFIRMED"


def test_parse_args_accepts_judgments_flag():
    from src.analysis.aggregate import parse_args

    args = parse_args(
        [
            "--questions", "q.jsonl",
            "--registry", "r.jsonl",
            "--generations", "g.jsonl",
            "--verifications", "v.jsonl",
            "--confidences", "c.jsonl",
            "--judgments", "j1.jsonl", "j2.jsonl",
            "--output-dir", "out",
        ]
    )
    assert args.judgments == [Path("j1.jsonl"), Path("j2.jsonl")]


def test_parse_args_accepts_multiple_generation_files():
    from src.analysis.aggregate import parse_args

    args = parse_args(
        [
            "--questions",
            "q.jsonl",
            "--registry",
            "r.jsonl",
            "--generations",
            "a.jsonl",
            "b.jsonl",
            "--verifications",
            "av.jsonl",
            "bv.jsonl",
            "--confidences",
            "ac.jsonl",
            "bc.jsonl",
            "--output-dir",
            "out",
        ]
    )
    assert args.generations == [Path("a.jsonl"), Path("b.jsonl")]
    assert args.verifications == [Path("av.jsonl"), Path("bv.jsonl")]
    assert args.confidences == [Path("ac.jsonl"), Path("bc.jsonl")]
    assert args.output_dir == Path("out")
