"""Contract tests for src/analysis/judge_era.py.

The J2a judge ran over ~2 months and its behaviour changed twice mid-collection.
The only record of WHEN a citation was judged is its position in the append-only
judgments JSONL, so era has to be derived from row order and then carried to the
generation level, which is where the registered `fabricated` outcome lives.

Two properties matter and are pinned here:

  1. row_index counts FILE POSITION among non-blank lines, so it stays consistent
     with the existing audit scripts (scripts/_judge_drift_*.py) even if a row is
     dropped for carrying an error.
  2. a generation whose citations straddle an era boundary has no well-defined
     era. It is labelled MIXED rather than silently assigned one, so the drift
     script can drop it AND report how many it dropped (docs/drift_interaction_prespec.md §4).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.judge_era import (
    DEGRADED_END,
    EARLY_END,
    MIXED_ERA,
    era_of_row_index,
    generation_era_rows,
    load_judgment_meta,
    main,
    parse_custom_id,
)


def _judgment(cid: str, search_count: int = 2, error: str | None = None) -> dict:
    return {
        "custom_id": cid,
        "judge_status": "REAL",
        "judge_reasoning": "r",
        "judge_evidence_url": "",
        "judge_search_count": search_count,
        "error": error,
    }


def _cid(qid: str = "academic_obscure_real_0100", cit: int = 0) -> str:
    return f"{qid}__allenai/Llama-3.1-Tulu-3-8B__citations_expected__t0__c{cit}"


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


# ---------- era boundaries ----------


def test_era_boundaries_are_the_empirical_block_edges() -> None:
    """Boundaries come from grounding metrics and are NOT tuned to the outcome."""
    assert EARLY_END == 33 * 500
    assert DEGRADED_END == 48 * 500


@pytest.mark.parametrize(
    "index,expected",
    [
        (0, "EARLY"),
        (16_499, "EARLY"),
        (16_500, "DEGRADED"),
        (23_999, "DEGRADED"),
        (24_000, "RECOVERED"),
        (31_979, "RECOVERED"),
    ],
)
def test_era_of_row_index(index: int, expected: str) -> None:
    assert era_of_row_index(index) == expected


# ---------- custom_id parsing ----------


def test_parse_custom_id_keeps_the_slash_in_model_id() -> None:
    """Phase C open-weights ids embed the model_id's '/', which must survive."""
    qid, model_id, condition, trial_id, citation_index = parse_custom_id(_cid())
    assert qid == "academic_obscure_real_0100"
    assert model_id == "allenai/Llama-3.1-Tulu-3-8B"
    assert condition == "citations_expected"
    assert trial_id == 0
    assert citation_index == 0


def test_parse_custom_id_returns_int_keys() -> None:
    """trial_id merges against generations, where it is an int, not 't0'."""
    _, _, _, trial_id, citation_index = parse_custom_id(_cid(cit=7))
    assert isinstance(trial_id, int)
    assert citation_index == 7


# ---------- meta loading ----------


def test_load_judgment_meta_assigns_era_from_file_position(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "j.jsonl", [_judgment(_cid(cit=i)) for i in range(3)])
    meta = load_judgment_meta(path)
    assert [m["row_index"] for m in meta] == [0, 1, 2]
    assert {m["era"] for m in meta} == {"EARLY"}


def test_load_judgment_meta_skips_error_rows_without_shifting_indices(
    tmp_path: Path,
) -> None:
    """An error row must not renumber the rows after it, or era assignment would
    silently disagree with the standing drift audit scripts."""
    path = _write_jsonl(
        tmp_path / "j.jsonl",
        [
            _judgment(_cid(cit=0)),
            _judgment(_cid(cit=1), error="429 rate limit"),
            _judgment(_cid(cit=2)),
        ],
    )
    meta = load_judgment_meta(path)
    assert [m["row_index"] for m in meta] == [0, 2]
    assert [m["citation_index"] for m in meta] == [0, 2]


def test_load_judgment_meta_carries_the_grounding_covariate(tmp_path: Path) -> None:
    path = _write_jsonl(tmp_path / "j.jsonl", [_judgment(_cid(), search_count=3)])
    assert load_judgment_meta(path)[0]["judge_search_count"] == 3


# ---------- generation-level rollup ----------


def test_generation_rows_collapse_citations_of_one_generation(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "j.jsonl", [_judgment(_cid(cit=i)) for i in range(4)]
    )
    rows = generation_era_rows(load_judgment_meta(path))
    assert len(rows) == 1
    assert rows[0]["n_judged"] == 4
    assert rows[0]["era"] == "EARLY"
    assert rows[0]["n_eras"] == 1


def test_generation_rows_separate_distinct_generations(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "j.jsonl",
        [_judgment(_cid(qid="academic_known_real_0001")), _judgment(_cid())],
    )
    rows = generation_era_rows(load_judgment_meta(path))
    assert len(rows) == 2


def test_generation_spanning_an_era_boundary_is_labelled_mixed() -> None:
    """Straddling generations get MIXED, never a silently-picked era."""
    meta = [
        {
            "row_index": EARLY_END - 1,
            "era": "EARLY",
            "question_id": "q",
            "model_id": "m",
            "condition": "neutral",
            "trial_id": 0,
            "citation_index": 0,
            "judge_search_count": 2,
        },
        {
            "row_index": EARLY_END,
            "era": "DEGRADED",
            "question_id": "q",
            "model_id": "m",
            "condition": "neutral",
            "trial_id": 0,
            "citation_index": 1,
            "judge_search_count": 2,
        },
    ]
    rows = generation_era_rows(meta)
    assert len(rows) == 1
    assert rows[0]["era"] == MIXED_ERA
    assert rows[0]["n_eras"] == 2


def test_generation_rows_summarise_grounding(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "j.jsonl",
        [
            _judgment(_cid(cit=0), search_count=0),
            _judgment(_cid(cit=1), search_count=3),
            _judgment(_cid(cit=2), search_count=3),
        ],
    )
    row = generation_era_rows(load_judgment_meta(path))[0]
    assert row["mean_search_count"] == pytest.approx(2.0)
    assert row["n_search_zero"] == 1


def test_generation_rows_carry_merge_keys_matching_the_adapter(tmp_path: Path) -> None:
    """These four keys are what E1/E2 and build_confirmatory_inputs join on."""
    path = _write_jsonl(tmp_path / "j.jsonl", [_judgment(_cid())])
    row = generation_era_rows(load_judgment_meta(path))[0]
    assert set(row) >= {"question_id", "condition", "model_id", "trial_id"}
    assert row["trial_id"] == 0


# ---------- CLI ----------


def test_cli_writes_one_row_per_generation(tmp_path: Path) -> None:
    src = _write_jsonl(
        tmp_path / "j.jsonl",
        [_judgment(_cid(cit=0)), _judgment(_cid(cit=1)), _judgment(_cid(qid="q2"))],
    )
    out = tmp_path / "era.jsonl"
    assert main(["--judgments", str(src), "--output", str(out)]) == 0
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert {r["n_judged"] for r in rows} == {1, 2}
