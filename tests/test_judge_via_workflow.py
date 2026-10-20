"""Tests for src/judge_via_workflow.py — the Path 7 (Workflow-tool / Max-OAuth)
J2a dispatcher that replaces the claude -p subprocess path after it was found
to bill the paid API (GH anthropics/claude-code #43333, #37686).

Methodology-load-bearing invariants:
- the prompt written to each per-citation file is BYTE-IDENTICAL to the
  registered v2 §6 prompt rendered by src.judge_via_subagents (so the §11
  sub-amendment's "same prompt" claim holds across the two dispatch batches)
- only successful judgments (valid §6 status) are persisted; failed agents
  leave their custom_id un-persisted so the resume pass re-judges them
- persisted rows use the EXACT canonical 6-key schema the subprocess path
  wrote, so downstream aggregation reads one uniform file
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.judge_via_subagents import build_judge_inputs
from src.judge_via_workflow import (
    JudgeWorkflowItem,
    build_remaining_items,
    chunk,
    make_workflow_args,
    records_from_workflow_results,
    persist_records,
    successful_custom_ids,
)


_CANONICAL_KEYS = {
    "custom_id",
    "judge_status",
    "judge_reasoning",
    "judge_evidence_url",
    "judge_search_count",
    "error",
}


@pytest.fixture
def tmp_files(tmp_path: Path):
    """Toy extraction+verification+questions set: one verifier-REAL (skipped
    by J2a), one verifier-FABRICATED, one verifier-UNVERIFIABLE → 2 judge
    inputs."""
    extractions = tmp_path / "extractions.jsonl"
    verifications = tmp_path / "verifications.jsonl"
    qmap_path = tmp_path / "questions.jsonl"

    ext_rows = [
        {
            "question_id": "academic_known_real_0001",
            "model_id": "claude-haiku-4-5-20251001",
            "condition": "citations_expected",
            "trial_id": 0,
            "output_excerpt": "Vaswani et al. proposed the Transformer in NeurIPS 2017.",
            "citations": [
                {"raw_span": "Vaswani et al. (2017)", "title": "Attention Is All You Need",
                 "authors": ["Vaswani"], "year": 2017, "venue": "NeurIPS", "doi": None,
                 "case_name": None, "reporter": None},
                {"raw_span": "Fake 2024", "title": "Nonexistent Paper",
                 "authors": ["Doe"], "year": 2024, "venue": None, "doi": None,
                 "case_name": None, "reporter": None},
            ],
        },
        {
            "question_id": "legal_unanswerable_0050",
            "model_id": "claude-haiku-4-5-20251001",
            "condition": "neutral",
            "trial_id": 0,
            "output_excerpt": "Per Doe v. Bar, 99 F.4th 1 (2099)...",
            "citations": [
                {"raw_span": "Doe v. Bar, 99 F.4th 1 (2099)", "title": None,
                 "case_name": "Doe v. Bar", "reporter": "99 F.4th 1", "year": 2099,
                 "authors": []},
            ],
        },
    ]
    ver_rows = [
        {"question_id": "academic_known_real_0001", "model_id": "claude-haiku-4-5-20251001",
         "condition": "citations_expected", "trial_id": 0, "n_citations": 2,
         "citation_statuses": ["REAL", "FABRICATED"]},
        {"question_id": "legal_unanswerable_0050", "model_id": "claude-haiku-4-5-20251001",
         "condition": "neutral", "trial_id": 0, "n_citations": 1,
         "citation_statuses": ["UNVERIFIABLE"]},
    ]
    extractions.write_text("\n".join(json.dumps(r) for r in ext_rows) + "\n", encoding="utf-8")
    verifications.write_text("\n".join(json.dumps(r) for r in ver_rows) + "\n", encoding="utf-8")
    qmap_path.write_text(
        json.dumps({"id": "academic_known_real_0001", "domain": "academic", "question": "Q"}) + "\n" +
        json.dumps({"id": "legal_unanswerable_0050", "domain": "legal", "question": "Q"}) + "\n",
        encoding="utf-8",
    )
    return extractions, verifications, qmap_path


# ----------------------------- successful_custom_ids ------------------------

def test_successful_custom_ids_counts_only_clean_rows(tmp_path: Path):
    out = tmp_path / "judgments.jsonl"
    out.write_text(
        json.dumps({"custom_id": "a", "judge_status": "REAL", "judge_reasoning": "x",
                    "judge_evidence_url": "", "judge_search_count": 1, "error": None}) + "\n" +
        json.dumps({"custom_id": "b", "judge_status": None, "judge_reasoning": "",
                    "judge_evidence_url": "", "judge_search_count": 0,
                    "error": "subprocess_failed_or_unparseable"}) + "\n" +
        json.dumps({"custom_id": "c", "judge_status": None, "judge_reasoning": "",
                    "judge_evidence_url": "", "judge_search_count": 0, "error": None}) + "\n",
        encoding="utf-8",
    )
    done = successful_custom_ids(out)
    assert done == {"a"}, "only the clean, status-bearing row counts as done"


def test_successful_custom_ids_empty_when_no_file(tmp_path: Path):
    assert successful_custom_ids(tmp_path / "missing.jsonl") == set()


# ----------------------------- build_remaining_items ------------------------

def test_build_remaining_items_skips_done_and_writes_prompt_files(tmp_files, tmp_path: Path):
    extractions, verifications, qmap_path = tmp_files
    out = tmp_path / "judgments.jsonl"
    prompts_dir = tmp_path / "prompts"

    # Pre-seed the FABRICATED citation as already judged.
    fab_cid = "academic_known_real_0001__claude-haiku-4-5-20251001__citations_expected__t0__c1"
    out.write_text(
        json.dumps({"custom_id": fab_cid, "judge_status": "FABRICATED", "judge_reasoning": "x",
                    "judge_evidence_url": "", "judge_search_count": 1, "error": None}) + "\n",
        encoding="utf-8",
    )

    items = build_remaining_items(extractions, verifications, qmap_path, out, prompts_dir)

    assert len(items) == 1
    assert items[0].custom_id == "legal_unanswerable_0050__claude-haiku-4-5-20251001__neutral__t0__c0"
    p = Path(items[0].prompt_path)
    assert p.exists()
    assert "§6 JUDGMENT TAXONOMY" in p.read_text(encoding="utf-8")


def test_build_remaining_items_prompt_file_byte_identical_to_render(tmp_files, tmp_path: Path):
    """The written prompt file MUST equal the registered prompt verbatim, so
    the two dispatch batches judge identical content."""
    extractions, verifications, qmap_path = tmp_files
    out = tmp_path / "judgments.jsonl"  # empty → both inputs remain
    prompts_dir = tmp_path / "prompts"

    items = build_remaining_items(extractions, verifications, qmap_path, out, prompts_dir)

    expected_by_cid = {i.custom_id: i.prompt for i in
                       build_judge_inputs(extractions, verifications, qmap_path)}
    for item in items:
        on_disk = Path(item.prompt_path).read_text(encoding="utf-8")
        assert on_disk == expected_by_cid[item.custom_id], \
            f"prompt drift for {item.custom_id}"


def test_build_remaining_items_idempotent(tmp_files, tmp_path: Path):
    extractions, verifications, qmap_path = tmp_files
    out = tmp_path / "judgments.jsonl"
    prompts_dir = tmp_path / "prompts"

    first = build_remaining_items(extractions, verifications, qmap_path, out, prompts_dir)
    second = build_remaining_items(extractions, verifications, qmap_path, out, prompts_dir)

    assert [i.custom_id for i in first] == [i.custom_id for i in second]
    for item in second:
        assert Path(item.prompt_path).exists()


def test_build_remaining_items_handles_slash_in_model_id(tmp_path: Path):
    """Phase C open-weights model_ids embed '/' (e.g. 'allenai/OLMo-2-1124-13B').
    The custom_id therefore contains '/', which must NOT leak into the prompt
    filename as a path separator (that raised FileNotFoundError pre-fix), while
    the item's custom_id must still carry the exact, slash-bearing join key."""
    extractions = tmp_path / "extractions.jsonl"
    verifications = tmp_path / "verifications.jsonl"
    qmap_path = tmp_path / "questions.jsonl"
    out = tmp_path / "judgments.jsonl"
    prompts_dir = tmp_path / "prompts"

    model_id = "allenai/OLMo-2-1124-13B-Instruct"
    ext_row = {
        "question_id": "legal_obscure_real_0445",
        "model_id": model_id,
        "condition": "length_matched",
        "trial_id": 0,
        "output_excerpt": "Per Doe v. Bar, 1 F.4th 1 (2099)...",
        "citations": [
            {"raw_span": "Doe v. Bar", "title": None, "case_name": "Doe v. Bar",
             "reporter": "1 F.4th 1", "year": 2099, "authors": []},
        ],
    }
    ver_row = {
        "question_id": "legal_obscure_real_0445", "model_id": model_id,
        "condition": "length_matched", "trial_id": 0, "n_citations": 1,
        "citation_statuses": ["FABRICATED"],
    }
    extractions.write_text(json.dumps(ext_row) + "\n", encoding="utf-8")
    verifications.write_text(json.dumps(ver_row) + "\n", encoding="utf-8")
    qmap_path.write_text(
        json.dumps({"id": "legal_obscure_real_0445", "domain": "legal", "question": "Q"}) + "\n",
        encoding="utf-8")

    items = build_remaining_items(extractions, verifications, qmap_path, out, prompts_dir)

    assert len(items) == 1
    # exact join key preserved, slash and all
    assert items[0].custom_id == (
        "legal_obscure_real_0445__allenai/OLMo-2-1124-13B-Instruct__length_matched__t0__c0")
    p = Path(items[0].prompt_path)
    assert p.exists(), "prompt file must be written (no FileNotFoundError)"
    # the file lives DIRECTLY in prompts_dir — no '/'-induced subdirectory
    assert p.parent == prompts_dir
    assert "/" not in p.name and "\\" not in p.name
    assert "§6 JUDGMENT TAXONOMY" in p.read_text(encoding="utf-8")


def test_prompt_filename_backward_compatible_for_slash_free_ids():
    """Phase B (Claude) custom_ids have no '/'; their filename must stay the
    plain '<custom_id>.txt' so existing prompt caches are unchanged."""
    from src.judge_via_workflow import _prompt_filename
    cid = "academic_known_real_0001__claude-haiku-4-5-20251001__neutral__t0__c0"
    assert _prompt_filename(cid) == f"{cid}.txt"


def test_prompt_filename_injective_for_slash_bearing_ids():
    """Distinct custom_ids must map to distinct files even after sanitization,
    so one citation's prompt never overwrites another's."""
    from src.judge_via_workflow import _prompt_filename
    a = "q__org/model-A__neutral__t0__c0"
    b = "q__org-model-A__neutral__t0__c0"   # sanitizes to the same stem as `a`
    assert _prompt_filename(a) != _prompt_filename(b)


# ----------------------------- chunk ----------------------------------------

def test_chunk_splits_with_remainder():
    assert chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_chunk_exact_multiple():
    assert chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_chunk_empty():
    assert chunk([], 3) == []


# ----------------------------- make_workflow_args ---------------------------

def test_make_workflow_args_shape():
    items = [
        JudgeWorkflowItem(custom_id="a", prompt_path="/tmp/a.txt"),
        JudgeWorkflowItem(custom_id="b", prompt_path="/tmp/b.txt"),
    ]
    blob = make_workflow_args(items)
    assert blob == {"items": [
        {"custom_id": "a", "prompt_path": "/tmp/a.txt"},
        {"custom_id": "b", "prompt_path": "/tmp/b.txt"},
    ]}


# ----------------------------- records_from_workflow_results ----------------

def test_records_from_workflow_results_keeps_valid_drops_failed():
    results = [
        {"custom_id": "a", "status": "REAL", "reasoning": "ok", "evidence_url": "https://x", "search_count": 2},
        {"custom_id": "b", "error": "null_verdict"},
        {"custom_id": "c", "status": "FABRICATED", "reasoning": "none", "evidence_url": "", "search_count": 3},
        {"custom_id": "d", "status": "NOTASTATUS", "reasoning": "bad", "evidence_url": "", "search_count": 0},
    ]
    records, failed = records_from_workflow_results(results)

    assert sorted(r["custom_id"] for r in records) == ["a", "c"]
    assert sorted(failed) == ["b", "d"]


def test_records_from_workflow_results_canonical_schema():
    results = [{"custom_id": "a", "status": "REAL", "reasoning": "ok",
                "evidence_url": "https://x", "search_count": 2}]
    records, _ = records_from_workflow_results(results)
    assert set(records[0].keys()) == _CANONICAL_KEYS
    assert records[0]["judge_status"] == "REAL"
    assert records[0]["judge_reasoning"] == "ok"
    assert records[0]["judge_evidence_url"] == "https://x"
    assert records[0]["judge_search_count"] == 2
    assert records[0]["error"] is None


def test_records_from_workflow_results_coerces_search_count_and_missing_fields():
    results = [{"custom_id": "a", "status": "UNVERIFIABLE"}]  # missing optional fields
    records, failed = records_from_workflow_results(results)
    assert failed == []
    assert records[0]["judge_reasoning"] == ""
    assert records[0]["judge_evidence_url"] == ""
    assert records[0]["judge_search_count"] == 0


# ----------------------------- persist_records ------------------------------

def test_persist_records_appends_without_clobber(tmp_path: Path):
    out = tmp_path / "judgments.jsonl"
    out.write_text(
        json.dumps({"custom_id": "pre", "judge_status": "REAL", "judge_reasoning": "x",
                    "judge_evidence_url": "", "judge_search_count": 0, "error": None}) + "\n",
        encoding="utf-8",
    )
    new_records = [
        {"custom_id": "a", "judge_status": "FABRICATED", "judge_reasoning": "y",
         "judge_evidence_url": "", "judge_search_count": 1, "error": None},
    ]
    persist_records(new_records, out)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert [r["custom_id"] for r in rows] == ["pre", "a"]


def test_persist_records_creates_file_when_absent(tmp_path: Path):
    out = tmp_path / "sub" / "judgments.jsonl"
    persist_records([{"custom_id": "a", "judge_status": "REAL", "judge_reasoning": "",
                      "judge_evidence_url": "", "judge_search_count": 0, "error": None}], out)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["custom_id"] == "a"
