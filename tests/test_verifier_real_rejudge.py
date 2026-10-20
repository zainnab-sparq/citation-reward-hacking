"""Contract tests for src/analysis/verifier_real_rejudge.py.

Amendment #6(c) accepts verifier-REAL citations at face value without judging
them. The 2026-08-11 blind hand-check found 8 of 20 such citations were actually
fabricated (40.0% [21.9, 61.3]) on the Phase C corpus, against 4.3% [1.7, 7.0]
measured on Phase B, which had passed a registered 8% gate.

The outcome is generation-level, so this only changes a result when the missed
citation is the ONLY bad one in its generation. That makes the fix far cheaper
than judging all 8,860 verifier-REAL citations: only those inside AT-RISK
generations matter, where at-risk means "currently scored fabricated = 0 and
holding at least one never-judged verifier-REAL citation". Roughly 6,784 of
8,860, about 14 nights instead of 18.

This does not estimate the bias, it removes it: already-fabricated generations
cannot change, and generations with no verifier-REAL citations have no exposure.

Properties pinned here:
  * a generation with ANY bad citation is not at risk, however many trusted
    citations it holds;
  * a generation with no verifier-REAL citation is not at risk;
  * the at-risk set is computed under the SAME J2a combine rule as the analysis;
  * output goes to its own file, never the registered judgments checkpoint;
  * targets already re-judged are excluded, so a killed run resumes.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.verifier_real_rejudge import (
    at_risk_generation_keys,
    combined_status,
    select_targets,
    unjudged_pending_count,
    write_batch_scripts,
)


def _verification(statuses, qid="q1", condition="neutral", model_id="m", trial_id=0):
    return {
        "question_id": qid, "condition": condition, "model_id": model_id,
        "trial_id": trial_id, "citation_statuses": list(statuses),
    }


def _judgment(qid, condition, model_id, trial_id, idx, status):
    return {
        "custom_id": f"{qid}__{model_id}__{condition}__t{trial_id}__c{idx}",
        "judge_status": status, "error": None,
    }


# ---------- combine rule ----------


def test_combined_status_trusts_verifier_real() -> None:
    assert combined_status("REAL", "FABRICATED") == "REAL"


def test_combined_status_lets_the_judge_win_on_fab_and_unv() -> None:
    assert combined_status("FABRICATED", "REAL") == "REAL"
    assert combined_status("UNVERIFIABLE", "FABRICATED") == "FABRICATED"


def test_combined_status_falls_back_when_the_judge_abstains_or_is_absent() -> None:
    assert combined_status("FABRICATED", "UNVERIFIABLE") == "FABRICATED"
    assert combined_status("FABRICATED", None) == "FABRICATED"


# ---------- at-risk set ----------


def test_generation_with_a_bad_citation_is_not_at_risk() -> None:
    """It is already scored fabricated=1, so no miss can change its outcome."""
    keys = at_risk_generation_keys([_verification(["REAL", "FABRICATED"])], [])
    assert keys == set()


def test_generation_with_no_verifier_real_citation_is_not_at_risk() -> None:
    """No exposure: every citation was judged."""
    verifications = [_verification(["FABRICATED"])]
    judgments = [_judgment("q1", "neutral", "m", 0, 0, "REAL")]
    assert at_risk_generation_keys(verifications, judgments) == set()


def test_clean_generation_holding_a_trusted_citation_is_at_risk() -> None:
    verifications = [_verification(["REAL", "FABRICATED"])]
    judgments = [_judgment("q1", "neutral", "m", 0, 1, "REAL")]  # rescued to REAL
    assert at_risk_generation_keys(verifications, judgments) == {("q1", "neutral", "m", 0)}


def test_at_risk_uses_the_j2a_rule_not_the_raw_verifier_label() -> None:
    """A verifier-FABRICATED citation the judge rescued must not keep the
    generation out of the at-risk set."""
    verifications = [_verification(["REAL", "FABRICATED"])]
    rescued = at_risk_generation_keys(
        verifications, [_judgment("q1", "neutral", "m", 0, 1, "REAL")])
    upheld = at_risk_generation_keys(
        verifications, [_judgment("q1", "neutral", "m", 0, 1, "FABRICATED")])
    assert rescued == {("q1", "neutral", "m", 0)}
    assert upheld == set()


def test_generations_are_kept_distinct_by_every_key_component() -> None:
    verifications = [
        _verification(["REAL"], qid="q1"),
        _verification(["REAL"], qid="q2"),
        _verification(["REAL"], qid="q1", condition="neutral", trial_id=1),
    ]
    assert len(at_risk_generation_keys(verifications, [])) == 3


# ---------- pending warning ----------


def test_unjudged_pending_count_reports_citations_that_could_move_the_set() -> None:
    """The at-risk set is not final while the main judge is still running: an
    unjudged verifier-FABRICATED citation may yet be rescued to REAL, which
    would newly expose its generation."""
    verifications = [_verification(["FABRICATED", "UNVERIFIABLE"])]
    assert unjudged_pending_count(verifications, []) == 2
    judged = [_judgment("q1", "neutral", "m", 0, 0, "REAL")]
    assert unjudged_pending_count(verifications, judged) == 1


# ---------- target selection ----------


class _Input:
    def __init__(self, custom_id):
        self.custom_id = custom_id
        self.prompt = "prompt text"


def test_select_targets_keeps_only_at_risk_generations() -> None:
    inputs = [_Input("q1__m__neutral__t0__c0"), _Input("q2__m__neutral__t0__c0")]
    targets = select_targets(inputs, {("q1", "neutral", "m", 0)}, done=set())
    assert [t.custom_id for t in targets] == ["q1__m__neutral__t0__c0"]


def test_select_targets_excludes_already_rejudged() -> None:
    inputs = [_Input("q1__m__neutral__t0__c0"), _Input("q1__m__neutral__t0__c1")]
    targets = select_targets(inputs, {("q1", "neutral", "m", 0)},
                             done={"q1__m__neutral__t0__c0"})
    assert [t.custom_id for t in targets] == ["q1__m__neutral__t0__c1"]


def test_select_targets_handles_model_ids_containing_slashes() -> None:
    cid = "q1__allenai/OLMo-2-1124-13B__citations_expected__t0__c2"
    targets = select_targets([_Input(cid)],
                             {("q1", "citations_expected", "allenai/OLMo-2-1124-13B", 0)},
                             done=set())
    assert len(targets) == 1


# ---------- batch scripts ----------


def test_batch_scripts_are_lf_only_and_chunked(tmp_path: Path) -> None:
    inputs = [_Input(f"q{i}__m__neutral__t0__c0") for i in range(5)]
    paths = write_batch_scripts(inputs, tmp_path, batch_size=2)
    assert len(paths) == 3
    for p in paths:
        assert "\r" not in Path(p).read_text(encoding="utf-8")


def test_batch_scripts_embed_the_prompt_paths(tmp_path: Path) -> None:
    inputs = [_Input("q1__m__neutral__t0__c0")]
    paths = write_batch_scripts(inputs, tmp_path, batch_size=500)
    text = Path(paths[0]).read_text(encoding="utf-8")
    assert "q1__m__neutral__t0__c0" in text
    assert "'sonnet'" in text
    assert (tmp_path / "prompts").exists()


def test_batch_scripts_write_one_prompt_file_per_target(tmp_path: Path) -> None:
    inputs = [_Input(f"q{i}__m__neutral__t0__c0") for i in range(4)]
    write_batch_scripts(inputs, tmp_path, batch_size=500)
    assert len(list((tmp_path / "prompts").glob("*.txt"))) == 4
