"""Tests for the src/run_judge_workflow.py orchestration CLI (prep/persist)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src import run_judge_workflow as rjw


@pytest.fixture
def toy_dataset(tmp_path: Path):
    extractions = tmp_path / "extractions.jsonl"
    verifications = tmp_path / "verifications.jsonl"
    questions = tmp_path / "questions.jsonl"

    ext_rows = [
        {"question_id": "academic_known_real_0001", "model_id": "claude-haiku-4-5-20251001",
         "condition": "neutral", "trial_id": 0, "output_excerpt": "ctx",
         "citations": [
             {"raw_span": "A", "title": "Real", "authors": ["X"], "year": 2017, "venue": "V",
              "doi": None, "case_name": None, "reporter": None},
             {"raw_span": "B", "title": "Fake", "authors": ["Y"], "year": 2024, "venue": None,
              "doi": None, "case_name": None, "reporter": None},
             {"raw_span": "C", "title": "Maybe", "authors": ["Z"], "year": 2019, "venue": "W",
              "doi": None, "case_name": None, "reporter": None},
         ]},
    ]
    ver_rows = [
        {"question_id": "academic_known_real_0001", "model_id": "claude-haiku-4-5-20251001",
         "condition": "neutral", "trial_id": 0, "n_citations": 3,
         "citation_statuses": ["REAL", "FABRICATED", "UNVERIFIABLE"]},
    ]
    extractions.write_text("\n".join(json.dumps(r) for r in ext_rows) + "\n", encoding="utf-8")
    verifications.write_text("\n".join(json.dumps(r) for r in ver_rows) + "\n", encoding="utf-8")
    questions.write_text(
        json.dumps({"id": "academic_known_real_0001", "domain": "academic", "question": "Q"}) + "\n",
        encoding="utf-8",
    )
    return extractions, verifications, questions


def _common_args(toy, tmp_path, **extra):
    extractions, verifications, questions = toy
    ns = type("NS", (), {})()
    ns.extractions = str(extractions)
    ns.verifications = str(verifications)
    ns.questions = str(questions)
    ns.output = str(tmp_path / "judgments.jsonl")
    ns.cache = str(tmp_path / "cache")
    for k, v in extra.items():
        setattr(ns, k, v)
    return ns


def test_prep_writes_batch_scripts_and_manifest(toy_dataset, tmp_path: Path, capsys):
    ns = _common_args(toy_dataset, tmp_path, batch_size=2)
    rc = rjw.cmd_prep(ns)
    assert rc == 0

    batches_dir = tmp_path / "cache" / "batches"
    scripts = sorted(batches_dir.glob("batch_*.js"))
    # 2 judge-eligible citations (FAB + UNV), batch_size=2 → 1 batch script.
    assert len(scripts) == 1
    manifest = json.loads((batches_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_items"] == 2
    assert len(manifest["scripts"]) == 1


def test_prep_generated_script_embeds_items_and_is_parseable(toy_dataset, tmp_path: Path):
    ns = _common_args(toy_dataset, tmp_path, batch_size=10)
    rjw.cmd_prep(ns)
    script = (tmp_path / "cache" / "batches" / "batch_000.js").read_text(encoding="utf-8")

    # The embedded items literal must be extractable + valid JSON.
    m = re.search(r"const items = (\[.*?\])\n", script, re.DOTALL)
    assert m, "items literal not found"
    items = json.loads(m.group(1))
    assert len(items) == 2
    assert all("custom_id" in it and "prompt_path" in it for it in items)
    # Methodology guard: the referenced prompt files exist and carry the §6 prompt.
    for it in items:
        p = Path(it["prompt_path"])
        assert p.exists()
        assert "§6 JUDGMENT TAXONOMY" in p.read_text(encoding="utf-8")
    assert "model: 'sonnet'" in script
    assert "WebSearch" in script


def test_prep_writes_batch_scripts_with_lf_only_newlines(toy_dataset, tmp_path: Path):
    """The Workflow permission handler rejects a scriptPath whose file carries
    CR bytes ("control characters that would be hidden in the approval
    dialog"), so on Windows a text-mode write with platform newlines produced
    an undispatchable script every night. The batch script must be LF-only
    regardless of platform."""
    ns = _common_args(toy_dataset, tmp_path, batch_size=10)
    rjw.cmd_prep(ns)
    raw = (tmp_path / "cache" / "batches" / "batch_000.js").read_bytes()
    assert b"\r" not in raw
    assert b"\n" in raw


def test_prep_reprep_clears_stale_batch_scripts(toy_dataset, tmp_path: Path):
    ns = _common_args(toy_dataset, tmp_path, batch_size=1)
    rjw.cmd_prep(ns)
    batches_dir = tmp_path / "cache" / "batches"
    first = sorted(p.name for p in batches_dir.glob("batch_*.js"))
    assert first == ["batch_000.js", "batch_001.js"]  # 2 items, size 1

    # Mark one as done, re-prep → fewer batches, no stale leftover.
    out = Path(ns.output)
    # find a custom_id from the manifest
    manifest = json.loads((batches_dir / "manifest.json").read_text(encoding="utf-8"))
    first_script = Path(manifest["scripts"][0]).read_text(encoding="utf-8")
    cid = re.search(r'"custom_id": "([^"]+)"', first_script).group(1)
    out.write_text(json.dumps({"custom_id": cid, "judge_status": "REAL", "judge_reasoning": "x",
                               "judge_evidence_url": "", "judge_search_count": 0, "error": None}) + "\n",
                   encoding="utf-8")
    rjw.cmd_prep(ns)
    second = sorted(p.name for p in batches_dir.glob("batch_*.js"))
    assert second == ["batch_000.js"], "stale batch_001.js must be cleared on re-prep"


def test_persist_round_trips_a_result_file(toy_dataset, tmp_path: Path):
    ns = _common_args(toy_dataset, tmp_path, result=str(tmp_path / "result.json"))
    result = {
        "batch": "j2a-batch-000",
        "n": 2,
        "results": [
            {"custom_id": "cid_a", "status": "REAL", "reasoning": "ok", "evidence_url": "https://x", "search_count": 1},
            {"custom_id": "cid_b", "error": "null_verdict"},
        ],
    }
    Path(ns.result).write_text(json.dumps(result), encoding="utf-8")
    rc = rjw.cmd_persist(ns)
    assert rc == 0

    rows = [json.loads(l) for l in Path(ns.output).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["custom_id"] == "cid_a"
    assert rows[0]["judge_status"] == "REAL"
    assert rows[0]["error"] is None


def test_persist_unwraps_workflow_envelope(toy_dataset, tmp_path: Path):
    """The Workflow task output file wraps the script's return value as
    {summary, agentCount, logs, result: {batch, n, results: [...]}}. persist
    must descend into .result to find the results array."""
    ns = _common_args(toy_dataset, tmp_path, result=str(tmp_path / "result.json"))
    envelope = {
        "summary": "Path 7 J2a judge batch ...",
        "agentCount": 1,
        "logs": ["..."],
        "result": {
            "batch": "j2a-batch-000",
            "n": 1,
            "results": [
                {"custom_id": "cid_a", "status": "REAL", "reasoning": "ok",
                 "evidence_url": "https://x", "search_count": 1},
            ],
        },
    }
    Path(ns.result).write_text(json.dumps(envelope), encoding="utf-8")
    assert rjw.cmd_persist(ns) == 0
    rows = [json.loads(l) for l in Path(ns.output).read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["custom_id"] == "cid_a"
    assert rows[0]["judge_status"] == "REAL"


def test_persist_accepts_bare_list_payload(toy_dataset, tmp_path: Path):
    ns = _common_args(toy_dataset, tmp_path, result=str(tmp_path / "result.json"))
    Path(ns.result).write_text(json.dumps([
        {"custom_id": "cid_a", "status": "FABRICATED", "reasoning": "no", "evidence_url": "", "search_count": 2},
    ]), encoding="utf-8")
    assert rjw.cmd_persist(ns) == 0
    rows = [json.loads(l) for l in Path(ns.output).read_text(encoding="utf-8").splitlines()]
    assert rows[0]["judge_status"] == "FABRICATED"
