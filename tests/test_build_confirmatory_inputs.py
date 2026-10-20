"""Contract tests for src/analysis/build_confirmatory_inputs.py.

The registered confirmatory scripts (e1_checkpoint_cascade.R,
e2_prompt_manipulation.R) read JSONL directly via `read_jsonl`, take a SINGLE
--generations file, merge the model registry on `family`, and require one
verification record per CITATION carrying `status`. Three things in the Phase C
tree do not match that contract:

  1. data/verifications/phase_c/verifications.jsonl is nested one record per
     GENERATION (`citation_statuses` list, no top-level `status`).
  2. The J2a judge verdicts live in a separate file and are not merged anywhere
     on the JSONL path, so E1/E2 would silently score raw verifier labels.
  3. data/models/phase_c_checkpoints.jsonl names the grouping column
     `model_family`, not `family`.

This module bridges those gaps WITHOUT touching the registered R scripts, and
reuses aggregate.py's already-tested flatten / combine_j2a logic so the J2a
rule has exactly one implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.build_confirmatory_inputs import (
    build_confirmatory_verifications,
    concat_generations,
    main,
    to_confirmatory_models,
    to_confirmatory_questions,
    write_jsonl,
)


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    return path


# ---------- model registry ----------


PHASE_C_RECORD = {
    "model_id": "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "model_family": "tulu3_8b",
    "stage": "SFT",
    "stage_ordinal": 1,
    "is_post_trained": 1,
    "is_open_weights": 1,
}

PHASE_B_RECORD = {
    "model_id": "claude-haiku-4-5-20251001",
    "family": "claude",
    "is_post_trained": 1,
    "is_open_weights": 0,
}


def test_models_maps_phase_c_model_family_to_family(tmp_path: Path) -> None:
    """E1 line 181 and E2 line 169 both merge on `family`."""
    src = _write_jsonl(tmp_path / "reg.jsonl", [PHASE_C_RECORD])
    rows = to_confirmatory_models(src)
    assert rows[0]["family"] == "tulu3_8b"
    assert "model_family" not in rows[0]


def test_models_preserves_phase_b_family_key(tmp_path: Path) -> None:
    """The Phase B registry already uses `family`; must not regress."""
    src = _write_jsonl(tmp_path / "reg.jsonl", [PHASE_B_RECORD])
    rows = to_confirmatory_models(src)
    assert rows[0]["family"] == "claude"


def test_models_carries_stage_ordinal(tmp_path: Path) -> None:
    """E1 needs stage_ordinal; aggregate.to_models_frame drops it."""
    src = _write_jsonl(tmp_path / "reg.jsonl", [PHASE_C_RECORD])
    assert to_confirmatory_models(src)[0]["stage_ordinal"] == 1


def test_models_stage_ordinal_is_none_when_registry_omits_it(
    tmp_path: Path,
) -> None:
    """Phase B has no cascade, so its registry has no stage_ordinal."""
    src = _write_jsonl(tmp_path / "reg.jsonl", [PHASE_B_RECORD])
    assert to_confirmatory_models(src)[0]["stage_ordinal"] is None


def test_models_rejects_registry_with_neither_family_key(
    tmp_path: Path,
) -> None:
    """A registry missing both spellings is a contract error, not a silent None."""
    src = _write_jsonl(tmp_path / "reg.jsonl", [{"model_id": "m", "is_post_trained": 1}])
    with pytest.raises(KeyError):
        to_confirmatory_models(src)


# ---------- verifications flatten + J2a merge ----------


def _verification(statuses: list[str]) -> dict:
    return {
        "question_id": "academic_obscure_real_0100",
        "condition": "citations_expected",
        "model_id": "allenai/Llama-3.1-Tulu-3-8B",
        "trial_id": 0,
        "n_citations": len(statuses),
        "citation_statuses": statuses,
    }


def _judgment(idx: int, judge_status: str | None, error: str | None = None) -> dict:
    return {
        "custom_id": (
            "academic_obscure_real_0100__allenai/Llama-3.1-Tulu-3-8B"
            f"__citations_expected__t0__c{idx}"
        ),
        "judge_status": judge_status,
        "error": error,
    }


def test_verifications_emit_one_row_per_citation(tmp_path: Path) -> None:
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL", "FABRICATED", "UNVERIFIABLE"])])
    rows = build_confirmatory_verifications([vpath], [])
    assert len(rows) == 3
    assert [r["citation_index"] for r in rows] == [0, 1, 2]
    assert all("status" in r for r in rows)


def test_verifier_real_is_mapped_to_real_confirmed(tmp_path: Path) -> None:
    """R's GOOD_STATUSES uses REAL_CONFIRMED, not REAL."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL"])])
    assert build_confirmatory_verifications([vpath], [])[0]["status"] == "REAL_CONFIRMED"


def test_judge_real_rescues_verifier_fabricated(tmp_path: Path) -> None:
    """The whole point of the J2a layer: rescuing false-FABRICATED calls."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["FABRICATED"])])
    jpath = _write_jsonl(tmp_path / "j.jsonl", [_judgment(0, "REAL")])
    row = build_confirmatory_verifications([vpath], [jpath])[0]
    assert row["status"] == "REAL_CONFIRMED"
    assert row["verifier_status"] == "FABRICATED"


def test_judge_unverifiable_falls_back_to_verifier_label(tmp_path: Path) -> None:
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["FABRICATED"])])
    jpath = _write_jsonl(tmp_path / "j.jsonl", [_judgment(0, "UNVERIFIABLE")])
    assert build_confirmatory_verifications([vpath], [jpath])[0]["status"] == "FABRICATED"


def test_errored_judgment_falls_back_to_verifier_label(tmp_path: Path) -> None:
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["UNVERIFIABLE"])])
    jpath = _write_jsonl(tmp_path / "j.jsonl", [_judgment(0, "REAL", error="429 rate limit")])
    assert build_confirmatory_verifications([vpath], [jpath])[0]["status"] == "UNVERIFIABLE"


def test_verifier_real_is_never_overridden_by_the_judge(tmp_path: Path) -> None:
    """Restricted firing: the judge only fires on verifier FAB/UNV."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL"])])
    jpath = _write_jsonl(tmp_path / "j.jsonl", [_judgment(0, "FABRICATED")])
    assert build_confirmatory_verifications([vpath], [jpath])[0]["status"] == "REAL_CONFIRMED"


def test_unjudged_citation_keeps_verifier_label(tmp_path: Path) -> None:
    """Partial judge coverage must not corrupt un-judged rows."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["FABRICATED", "FABRICATED"])])
    jpath = _write_jsonl(tmp_path / "j.jsonl", [_judgment(0, "REAL")])
    rows = build_confirmatory_verifications([vpath], [jpath])
    assert rows[0]["status"] == "REAL_CONFIRMED"
    assert rows[1]["status"] == "FABRICATED"


# ---------- generations concat ----------


def test_concat_generations_merges_files_and_drops_error_rows(
    tmp_path: Path,
) -> None:
    """E1/E2 take ONE --generations file; Phase C has twelve."""
    a = _write_jsonl(
        tmp_path / "a.jsonl",
        [{"question_id": "q1", "condition": "neutral", "model_id": "m1",
          "trial_id": 0, "output_text": "x", "error": None}],
    )
    b = _write_jsonl(
        tmp_path / "b.jsonl",
        [{"question_id": "q2", "condition": "neutral", "model_id": "m2",
          "trial_id": 0, "output_text": "y", "error": None},
         {"question_id": "q3", "condition": "neutral", "model_id": "m2",
          "trial_id": 0, "output_text": None, "error": "timeout"}],
    )
    rows = concat_generations([a, b])
    assert [r["question_id"] for r in rows] == ["q1", "q2"]


# ---------- questions ----------


def test_questions_drop_nested_fields(tmp_path: Path) -> None:
    """The frozen question file carries nested `expected_citation` / `metadata`
    dicts. E1/E2 build a one-row data.frame PER LINE and rbind them, and a
    nested object makes `as.data.frame` raise "arguments imply differing number
    of rows". Projecting to the three columns E1/E2 actually merge on is what
    makes the file readable at all."""
    src = _write_jsonl(
        tmp_path / "q.jsonl",
        [{
            "id": "academic_known_real_0000",
            "cell": "known_real",
            "domain": "academic",
            "question": "...",
            "expected_citation": {"authors": ["A"], "cited_by_count": 10789},
            "metadata": {"filter": "x"},
        }],
    )
    rows = to_confirmatory_questions(src)
    assert rows == [{
        "id": "academic_known_real_0000",
        "cell": "known_real",
        "domain": "academic",
    }]


# ---------- round trip ----------


# ---------- verifier-REAL re-judge correction (opt-in) ----------
#
# Amendment #6(c) trusts verifier-REAL citations without judging them. The blind
# hand-check measured that trust step at 60% correct on Phase C (8 of 20 were
# actually fabricated), against 4.3% disagreement on Phase B under a registered
# 8% gate. The correction is applied ONLY when its file is passed, so the
# registered pipeline stays the default and the deviation is explicit.


def _rejudgment(idx: int, status: str, error: str | None = None) -> dict:
    return {
        "custom_id": (
            "academic_obscure_real_0100__allenai/Llama-3.1-Tulu-3-8B"
            f"__citations_expected__t0__c{idx}"
        ),
        "judge_status": status,
        "error": error,
    }


def test_verifier_real_rejudgments_are_not_applied_by_default(tmp_path: Path) -> None:
    """The registered combine rule is the default; the correction is opt-in."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL"])])
    rows = build_confirmatory_verifications([vpath], [])
    assert rows[0]["status"] == "REAL_CONFIRMED"
    assert rows[0]["verifier_real_rejudged"] is False


def test_rejudged_fabricated_overrides_a_trusted_verifier_real(tmp_path: Path) -> None:
    """The whole point: a citation the verifier wrongly trusted becomes bad."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL"])])
    rpath = _write_jsonl(tmp_path / "r.jsonl", [_rejudgment(0, "FABRICATED")])
    rows = build_confirmatory_verifications([vpath], [], [rpath])
    assert rows[0]["status"] == "FABRICATED"
    assert rows[0]["verifier_status"] == "REAL_CONFIRMED"
    assert rows[0]["verifier_real_rejudged"] is True


def test_rejudged_real_confirms_the_verifier(tmp_path: Path) -> None:
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL"])])
    rpath = _write_jsonl(tmp_path / "r.jsonl", [_rejudgment(0, "REAL")])
    rows = build_confirmatory_verifications([vpath], [], [rpath])
    assert rows[0]["status"] == "REAL_CONFIRMED"
    assert rows[0]["verifier_real_rejudged"] is True


def test_rejudged_unverifiable_leaves_the_verifier_label_standing(tmp_path: Path) -> None:
    """Same fallback semantics as the main judge: abstention is not evidence."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL"])])
    rpath = _write_jsonl(tmp_path / "r.jsonl", [_rejudgment(0, "UNVERIFIABLE")])
    assert build_confirmatory_verifications([vpath], [], [rpath])[0]["status"] == "REAL_CONFIRMED"


def test_errored_rejudgment_leaves_the_verifier_label_standing(tmp_path: Path) -> None:
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL"])])
    rpath = _write_jsonl(tmp_path / "r.jsonl", [_rejudgment(0, "FABRICATED", error="429")])
    assert build_confirmatory_verifications([vpath], [], [rpath])[0]["status"] == "REAL_CONFIRMED"


def test_rejudgment_never_touches_a_citation_the_main_judge_owns(tmp_path: Path) -> None:
    """Verifier-FAB/UNV citations are the main judge's territory. A stray
    re-judgment on one must not double-override its verdict."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["FABRICATED"])])
    jpath = _write_jsonl(tmp_path / "j.jsonl", [_judgment(0, "REAL")])
    rpath = _write_jsonl(tmp_path / "r.jsonl", [_rejudgment(0, "FABRICATED")])
    rows = build_confirmatory_verifications([vpath], [jpath], [rpath])
    assert rows[0]["status"] == "REAL_CONFIRMED"     # the main judge's rescue stands
    assert rows[0]["verifier_real_rejudged"] is False


def test_unrejudged_trusted_citations_are_flagged_false(tmp_path: Path) -> None:
    """Partial coverage must be distinguishable from full coverage."""
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL", "REAL"])])
    rpath = _write_jsonl(tmp_path / "r.jsonl", [_rejudgment(0, "FABRICATED")])
    rows = build_confirmatory_verifications([vpath], [], [rpath])
    assert [r["verifier_real_rejudged"] for r in rows] == [True, False]


def test_cli_accepts_the_rejudgment_flag(tmp_path: Path) -> None:
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["REAL"])])
    rpath = _write_jsonl(tmp_path / "r.jsonl", [_rejudgment(0, "FABRICATED")])
    reg = _write_jsonl(tmp_path / "reg.jsonl", [PHASE_C_RECORD])
    gen = _write_jsonl(
        tmp_path / "g.jsonl",
        [{"question_id": "academic_obscure_real_0100", "condition": "citations_expected",
          "model_id": "allenai/Llama-3.1-Tulu-3-8B", "trial_id": 0,
          "output_text": "x", "error": None}],
    )
    q = _write_jsonl(tmp_path / "q.jsonl",
                     [{"id": "academic_obscure_real_0100", "cell": "obscure_real",
                       "domain": "academic"}])
    outdir = tmp_path / "out"
    rc = main([
        "--verifications", str(vpath), "--registry", str(reg),
        "--generations", str(gen), "--questions", str(q),
        "--verifier-real-rejudgments", str(rpath),
        "--output-dir", str(outdir),
    ])
    assert rc == 0
    rows = [json.loads(l) for l in (outdir / "verifications.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["status"] == "FABRICATED"


def test_write_jsonl_round_trips(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    write_jsonl([{"a": 1}, {"a": 2}], out)
    assert [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()] == [
        {"a": 1},
        {"a": 2},
    ]


def test_cli_writes_all_four_files(tmp_path: Path) -> None:
    vpath = _write_jsonl(tmp_path / "v.jsonl", [_verification(["FABRICATED"])])
    jpath = _write_jsonl(tmp_path / "j.jsonl", [_judgment(0, "REAL")])
    reg = _write_jsonl(tmp_path / "reg.jsonl", [PHASE_C_RECORD])
    gen = _write_jsonl(
        tmp_path / "g.jsonl",
        [{"question_id": "q1", "condition": "neutral", "model_id": "m1",
          "trial_id": 0, "output_text": "x", "error": None}],
    )
    q = _write_jsonl(
        tmp_path / "q.jsonl",
        [{"id": "q1", "cell": "known_real", "domain": "academic",
          "metadata": {"a": 1}}],
    )
    outdir = tmp_path / "out"
    rc = main([
        "--verifications", str(vpath),
        "--judgments", str(jpath),
        "--registry", str(reg),
        "--generations", str(gen),
        "--questions", str(q),
        "--output-dir", str(outdir),
    ])
    assert rc == 0
    questions = [json.loads(l) for l in (outdir / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert questions == [{"id": "q1", "cell": "known_real", "domain": "academic"}]
    verifications = [json.loads(l) for l in (outdir / "verifications.jsonl").read_text(encoding="utf-8").splitlines()]
    assert verifications[0]["status"] == "REAL_CONFIRMED"
    models = [json.loads(l) for l in (outdir / "models.jsonl").read_text(encoding="utf-8").splitlines()]
    assert models[0]["family"] == "tulu3_8b"
    generations = [json.loads(l) for l in (outdir / "generations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(generations) == 1
