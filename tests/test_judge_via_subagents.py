"""Tests for src/judge_via_subagents.py — the headless `claude -p` subprocess
dispatcher for the J2a judge layer (amendment #6 (d) operational pivot).

The methodology-load-bearing pieces:
- input filtering fires only on verifier-FAB / verifier-UNV (J2a rule)
- prompt template matches the existing Step 0 calibrated v2 §6 prompt
- subprocess output JSON is parsed losslessly (status, reasoning, evidence_url, search_count)
- resume-on-restart skips already-judged custom_ids
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.judge_via_subagents import (
    JudgeInput,
    JudgeRecord,
    TransientSubprocessError,
    build_judge_inputs,
    dispatch_one,
    parse_subprocess_output,
    run_dispatch_sync,
)


@pytest.fixture
def tmp_files(tmp_path: Path):
    """Three-row toy extraction + verification dataset."""
    extractions = tmp_path / "extractions.jsonl"
    verifications = tmp_path / "verifications.jsonl"
    qmap_path = tmp_path / "questions.jsonl"

    ext_rows = [
        {
            "question_id": "academic_known_real_0001",
            "model_id": "claude-haiku-4-5-20251001",
            "condition": "citations_expected",
            "trial_id": 0,
            "extractor_model_id": "claude-haiku-4-5-20251001",
            "output_excerpt": "Vaswani et al. proposed the Transformer in NeurIPS 2017.",
            "citations": [
                {
                    "raw_span": "Vaswani et al. (2017)",
                    "title": "Attention Is All You Need",
                    "authors": ["Vaswani"],
                    "year": 2017,
                    "venue": "NeurIPS",
                    "doi": None,
                    "case_name": None,
                    "reporter": None,
                },
                {
                    "raw_span": "Fake 2024",
                    "title": "Nonexistent Paper",
                    "authors": ["Doe"],
                    "year": 2024,
                    "venue": None,
                    "doi": None,
                    "case_name": None,
                    "reporter": None,
                },
            ],
        },
        {
            "question_id": "legal_unanswerable_0050",
            "model_id": "claude-haiku-4-5-20251001",
            "condition": "neutral",
            "trial_id": 0,
            "extractor_model_id": "claude-haiku-4-5-20251001",
            "output_excerpt": "Per Doe v. Bar, 99 F.4th 1 (2099)...",
            "citations": [
                {
                    "raw_span": "Doe v. Bar, 99 F.4th 1 (2099)",
                    "title": None,
                    "case_name": "Doe v. Bar",
                    "reporter": "99 F.4th 1",
                    "year": 2099,
                    "authors": [],
                }
            ],
        },
    ]
    ver_rows = [
        {
            "question_id": "academic_known_real_0001",
            "model_id": "claude-haiku-4-5-20251001",
            "condition": "citations_expected",
            "trial_id": 0,
            "n_citations": 2,
            "citation_statuses": ["REAL", "FABRICATED"],
        },
        {
            "question_id": "legal_unanswerable_0050",
            "model_id": "claude-haiku-4-5-20251001",
            "condition": "neutral",
            "trial_id": 0,
            "n_citations": 1,
            "citation_statuses": ["UNVERIFIABLE"],
        },
    ]
    extractions.write_text(
        "\n".join(json.dumps(r) for r in ext_rows) + "\n", encoding="utf-8"
    )
    verifications.write_text(
        "\n".join(json.dumps(r) for r in ver_rows) + "\n", encoding="utf-8"
    )
    qmap_path.write_text(
        json.dumps({"id": "academic_known_real_0001", "domain": "academic", "question": "Q", "expected_existence": "answerable"}) + "\n" +
        json.dumps({"id": "legal_unanswerable_0050", "domain": "legal", "question": "Q", "expected_existence": "unanswerable"}) + "\n",
        encoding="utf-8",
    )
    return extractions, verifications, qmap_path


def test_build_judge_inputs_fires_only_on_verifier_fab_or_unv(tmp_files):
    """J2a rule: verifier-REAL citations skip the judge; verifier-FAB and
    verifier-UNV citations get judged. The 3-citation toy set has
    statuses [REAL, FABRICATED, UNVERIFIABLE] → 2 judge calls."""
    extractions, verifications, qmap_path = tmp_files

    inputs = list(build_judge_inputs(extractions, verifications, qmap_path))

    assert len(inputs) == 2
    statuses_judged = sorted(i.verifier_status for i in inputs)
    assert statuses_judged == ["FABRICATED", "UNVERIFIABLE"]


def test_build_judge_inputs_custom_id_round_trips(tmp_files):
    """custom_id includes the citation_index suffix so judgments key back
    to the exact citation. Format: `<qid>__<model>__<cond>__t<trial>__c<cidx>`."""
    extractions, verifications, qmap_path = tmp_files

    inputs = list(build_judge_inputs(extractions, verifications, qmap_path))
    fab = next(i for i in inputs if i.verifier_status == "FABRICATED")

    assert fab.question_id == "academic_known_real_0001"
    assert fab.citation_index == 1
    assert "c1" in fab.custom_id
    assert "academic_known_real_0001" in fab.custom_id


def test_build_judge_inputs_renders_prompt_with_v2_taxonomy(tmp_files):
    """The prompt MUST contain the v2 §6 taxonomy markers so the calibrated
    Step 0 96% gate carries forward. Methodology drift here would invalidate
    amendment #6 (d) calibration."""
    extractions, verifications, qmap_path = tmp_files

    inputs = list(build_judge_inputs(extractions, verifications, qmap_path))

    for inp in inputs:
        assert "§6 JUDGMENT TAXONOMY" in inp.prompt
        assert "EXISTENT_WRONG_ID" in inp.prompt
        assert "metadata error" in inp.prompt
        assert "REAL, FABRICATED, or UNVERIFIABLE" in inp.prompt


def test_parse_subprocess_output_extracts_final_json_line():
    """The model may wrap output in a code fence or trail prose. The parser
    finds the last `{...}` block containing a `status` key and parses it."""
    raw = (
        "Some preamble.\n"
        '```json\n{"status": "REAL", "reasoning": "ok", "evidence_url": "https://x", "search_count": 2}\n```\n'
        "trailing prose\n"
    )
    parsed = parse_subprocess_output(raw)

    assert parsed["status"] == "REAL"
    assert parsed["reasoning"] == "ok"
    assert parsed["evidence_url"] == "https://x"
    assert parsed["search_count"] == 2


def test_parse_subprocess_output_handles_inline_final_json():
    """Bare JSON line (no fence) is also valid output per the prompt spec."""
    raw = 'prose\n{"status": "FABRICATED", "reasoning": "no work matches", "evidence_url": "", "search_count": 3}\n'
    parsed = parse_subprocess_output(raw)

    assert parsed["status"] == "FABRICATED"


def test_parse_subprocess_output_returns_none_on_unparseable():
    """Malformed output yields None so the caller can record an error
    instead of crashing the dispatch loop."""
    raw = "I refuse to answer."
    assert parse_subprocess_output(raw) is None


def test_run_dispatch_sync_writes_one_record_per_input(tmp_path: Path):
    """Each JudgeInput produces exactly one JudgeRecord written to the JSONL."""
    inputs = [
        JudgeInput(
            custom_id=f"acad__h45d__ce__t0__c{i}",
            question_id=f"qid_{i}",
            model_snapshot="claude-haiku-4-5-20251001",
            condition="ce",
            trial_id=0,
            citation_index=i,
            verifier_status="FABRICATED",
            prompt=f"Judge prompt for cite {i}",
        )
        for i in range(3)
    ]
    out = tmp_path / "judgments.jsonl"

    def fake_dispatch(prompt: str, *, model: str, working_dir: Path, timeout_s: int) -> dict | None:
        return {
            "status": "FABRICATED",
            "reasoning": f"reason for {prompt[-1]}",
            "evidence_url": "",
            "search_count": 1,
        }

    with patch("src.judge_via_subagents.dispatch_one", side_effect=fake_dispatch):
        run_dispatch_sync(inputs, out, workers=2, working_dir=tmp_path)

    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 3
    assert all(r["judge_status"] == "FABRICATED" for r in records)
    assert sorted(r["custom_id"] for r in records) == [
        "acad__h45d__ce__t0__c0",
        "acad__h45d__ce__t0__c1",
        "acad__h45d__ce__t0__c2",
    ]


def test_run_dispatch_sync_resumes_skipping_already_judged(tmp_path: Path):
    """Restart safety: if the output file already contains records for some
    custom_ids, those are NOT re-dispatched. This is what makes the 7+ hour
    job restart-safe across Max-quota throttle pauses."""
    out = tmp_path / "judgments.jsonl"
    out.write_text(
        json.dumps({
            "custom_id": "acad__h45d__ce__t0__c0",
            "judge_status": "REAL",
            "judge_reasoning": "prior run",
            "judge_evidence_url": "",
            "judge_search_count": 0,
            "error": None,
        }) + "\n",
        encoding="utf-8",
    )
    inputs = [
        JudgeInput(
            custom_id="acad__h45d__ce__t0__c0",
            question_id="qid_0", model_snapshot="claude-haiku-4-5-20251001",
            condition="ce", trial_id=0, citation_index=0, verifier_status="FABRICATED",
            prompt="prompt 0",
        ),
        JudgeInput(
            custom_id="acad__h45d__ce__t0__c1",
            question_id="qid_1", model_snapshot="claude-haiku-4-5-20251001",
            condition="ce", trial_id=0, citation_index=1, verifier_status="FABRICATED",
            prompt="prompt 1",
        ),
    ]

    dispatched: list[str] = []

    def fake_dispatch(prompt: str, *, model: str, working_dir: Path, timeout_s: int) -> dict | None:
        dispatched.append(prompt)
        return {"status": "FABRICATED", "reasoning": "x", "evidence_url": "", "search_count": 1}

    with patch("src.judge_via_subagents.dispatch_one", side_effect=fake_dispatch):
        run_dispatch_sync(inputs, out, workers=1, working_dir=tmp_path)

    assert dispatched == ["prompt 1"], "must NOT redispatch prompt 0 (already judged)"
    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2  # original + new


def test_run_dispatch_sync_records_error_on_subprocess_failure(tmp_path: Path):
    """If dispatch_one returns None (subprocess failed / output unparseable),
    the JSONL still gets a row with `judge_status=None` + `error` populated
    so a re-run can pick up just the failed rows."""
    inputs = [
        JudgeInput(
            custom_id="acad__h45d__ce__t0__c0",
            question_id="qid_0", model_snapshot="claude-haiku-4-5-20251001",
            condition="ce", trial_id=0, citation_index=0, verifier_status="FABRICATED",
            prompt="prompt 0",
        ),
    ]
    out = tmp_path / "judgments.jsonl"

    with patch("src.judge_via_subagents.dispatch_one", return_value=None):
        run_dispatch_sync(inputs, out, workers=1, working_dir=tmp_path)

    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1
    assert records[0]["judge_status"] is None
    assert records[0]["error"] is not None


def test_dispatch_one_strips_api_key_env_for_max_oauth_routing():
    """The judge layer is registered (amendment #6 (d)) as Max-pool routed,
    NOT paid-API. Without an explicit env strip, ANTHROPIC_API_KEY leaks
    from dotenv into the claude -p subprocess and the subprocess bills
    against the paid API. dispatch_one MUST pass env= to subprocess.run
    with both ANTHROPIC_API_KEY and ANTHROPIC_AUTH_TOKEN removed so the
    subprocess falls back to OAuth (~/.claude/.credentials.json)."""
    captured: dict = {}

    def fake_run(cmd, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "type": "result", "subtype": "success", "is_error": False,
                "result": '{"status": "REAL", "reasoning": "ok", "evidence_url": "", "search_count": 0}',
            }),
            stderr="",
        )

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-fake", "ANTHROPIC_AUTH_TOKEN": "t", "PATH": "/usr/bin"}, clear=True):
        with patch("src.judge_via_subagents.subprocess.run", side_effect=fake_run):
            dispatch_one("any prompt", working_dir=Path("."), timeout_s=10)

    env = captured.get("env")
    assert env is not None, "dispatch_one must pass env= to subprocess.run, not inherit parent env"
    assert "ANTHROPIC_API_KEY" not in env, "ANTHROPIC_API_KEY must be stripped to route via Max OAuth"
    assert "ANTHROPIC_AUTH_TOKEN" not in env, "ANTHROPIC_AUTH_TOKEN must be stripped to route via Max OAuth"
    assert env.get("PATH") == "/usr/bin", "non-auth env vars must be preserved so claude binary is findable"


def test_dispatch_one_raises_transient_on_credit_balance_too_low():
    """If `claude -p` returns an envelope with is_error=true and
    result containing 'Credit balance is too low' (Max-pool rolling-window
    exhaustion under amendment #6 (d)), dispatch_one MUST raise
    TransientSubprocessError so the runner can leave the row unwritten and
    pick it up on the next resume after the window refreshes."""
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": True,
        "api_error_status": 400,
        "result": "Credit balance is too low",
        "duration_ms": 200,
    }
    completed = SimpleNamespace(
        returncode=1,
        stdout=json.dumps(envelope),
        stderr="",
    )
    with patch("src.judge_via_subagents.subprocess.run", return_value=completed):
        with pytest.raises(TransientSubprocessError):
            dispatch_one("any prompt", working_dir=Path("."), timeout_s=10)


def test_run_dispatch_sync_skips_writing_on_transient_error(tmp_path: Path):
    """When dispatch_one raises TransientSubprocessError (Max pool out of
    credits), the runner must NOT write a record. Leaving the custom_id
    unwritten is what lets the existing resume logic pick it up on the next
    restart after the rolling window refreshes."""
    inputs = [
        JudgeInput(
            custom_id="acad__h45d__ce__t0__c0",
            question_id="qid_0", model_snapshot="claude-haiku-4-5-20251001",
            condition="ce", trial_id=0, citation_index=0, verifier_status="FABRICATED",
            prompt="prompt 0",
        ),
        JudgeInput(
            custom_id="acad__h45d__ce__t0__c1",
            question_id="qid_1", model_snapshot="claude-haiku-4-5-20251001",
            condition="ce", trial_id=0, citation_index=1, verifier_status="FABRICATED",
            prompt="prompt 1",
        ),
    ]
    out = tmp_path / "judgments.jsonl"

    def fake_dispatch(prompt: str, *, model: str, working_dir: Path, timeout_s: int):
        if "prompt 0" in prompt:
            return {"status": "FABRICATED", "reasoning": "ok", "evidence_url": "", "search_count": 1}
        raise TransientSubprocessError("credit balance too low")

    with patch("src.judge_via_subagents.dispatch_one", side_effect=fake_dispatch):
        run_dispatch_sync(inputs, out, workers=1, working_dir=tmp_path)

    records = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 1, "transient-failed row must NOT be persisted"
    assert records[0]["custom_id"] == "acad__h45d__ce__t0__c0"
    assert records[0]["error"] is None
