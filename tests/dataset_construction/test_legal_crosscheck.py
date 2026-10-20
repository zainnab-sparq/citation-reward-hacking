"""Tests for the HARD legal-unanswerable cross-check gate (pre-reg §8).

The 100 legal unanswerable items use facially-impossible reporter citations
(e.g. "999 U.S. 22187"). Before any legal fabrication rate is reported, each
must be confirmed NON-existent against CourtListener's Citation Lookup API
(`POST /api/rest/v4/citation-lookup/`). A *match* is the bad outcome here: it
means a constructed "unanswerable" is actually a real case. That is the inverse
of normal citation verification, where a match is the good outcome.
"""
import json

import httpx
import pytest
import respx

from src.dataset_construction.legal_crosscheck import (
    CITATION_LOOKUP_URL,
    CourtListenerTokenMissing,
    CrosscheckRecord,
    GateState,
    LegalUnanswerable,
    Verdict,
    align_results,
    build_lookup_payload,
    classify_result,
    evaluate_gate,
    load_legal_unanswerables,
    run_crosscheck,
    write_crosscheck_log,
)


async def _noop_sleep(_seconds: float) -> None:
    return None


# ---- load_legal_unanswerables -------------------------------------------------

def test_load_filters_to_legal_unanswerable_only(tmp_path):
    rows = [
        {"domain": "legal", "cell": "unanswerable", "id": "legal_unanswerable_0",
         "expected_citation": {"reporter": "999 U.S. 22187", "case_name": "Chen v. DHS"}},
        {"domain": "legal", "cell": "known_real", "id": "legal_known_real_0",
         "expected_citation": {"reporter": "384 U.S. 436", "case_name": "Miranda v. Arizona"}},
        {"domain": "academic", "cell": "unanswerable", "id": "academic_unanswerable_0",
         "expected_citation": {"title": "Foo"}},
    ]
    p = tmp_path / "frozen.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    items = load_legal_unanswerables(p)

    assert len(items) == 1
    assert items[0].id == "legal_unanswerable_0"
    assert items[0].reporter == "999 U.S. 22187"
    assert items[0].case_name == "Chen v. DHS"


# ---- classify_result ----------------------------------------------------------

def test_classify_real_match_when_clusters_present():
    result = {"status": 200, "clusters": [{"id": 123, "case_name": "Real Case"}]}
    assert classify_result(result) == Verdict.REAL_MATCH


def test_classify_real_match_on_multiple_matches_status_300():
    result = {"status": 300, "clusters": [{"id": 1}, {"id": 2}]}
    assert classify_result(result) == Verdict.REAL_MATCH


def test_classify_no_match_when_status_404_and_no_clusters():
    result = {"status": 404, "clusters": [], "error_message": "not found"}
    assert classify_result(result) == Verdict.NO_MATCH


def test_classify_no_match_when_status_400_invalid_reporter():
    result = {"status": 400, "clusters": [], "error_message": "reporter not found"}
    assert classify_result(result) == Verdict.NO_MATCH


def test_classify_unverifiable_on_rate_limit_429():
    result = {"status": 429, "clusters": []}
    assert classify_result(result) == Verdict.UNVERIFIABLE


def test_classify_unverifiable_when_status_200_but_no_clusters():
    # Contradictory: 200 but nothing matched -> cannot certify either way.
    result = {"status": 200, "clusters": []}
    assert classify_result(result) == Verdict.UNVERIFIABLE


# ---- build_lookup_payload -----------------------------------------------------

def test_build_lookup_payload_joins_reporters_as_text():
    payload = build_lookup_payload(["999 U.S. 22187", "12 F.4th 8847"])
    assert payload == {"text": "999 U.S. 22187\n12 F.4th 8847"}


# ---- align_results ------------------------------------------------------------

def test_align_matches_results_to_records_by_normalized_citation():
    records = [
        LegalUnanswerable(id="a", reporter="999 U.S. 22187", case_name="Chen v. DHS"),
        LegalUnanswerable(id="b", reporter="12 F.4th 8847", case_name="Doe v. Roe"),
    ]
    # API returns them out of order and with extra whitespace in the citation.
    results = [
        {"citation": "12  F.4th  8847", "normalized_citations": ["12 F.4th 8847"],
         "status": 200, "clusters": [{"id": 5}]},
        {"citation": "999 U.S. 22187", "normalized_citations": ["999 U.S. 22187"],
         "status": 404, "clusters": []},
    ]
    out = align_results(records, results)
    by_id = {r.id: r for r in out}
    assert by_id["a"].verdict == Verdict.NO_MATCH
    assert by_id["b"].verdict == Verdict.REAL_MATCH
    assert by_id["b"].cluster_ids == [5]


def test_align_marks_unverifiable_when_no_result_for_record():
    records = [LegalUnanswerable(id="a", reporter="999 U.S. 22187", case_name="Chen")]
    out = align_results(records, [])  # eyecite returned nothing for this citation
    assert out[0].verdict == Verdict.UNVERIFIABLE


# ---- evaluate_gate ------------------------------------------------------------

def _rec(verdict: Verdict, rec_id: str = "x") -> CrosscheckRecord:
    return CrosscheckRecord(id=rec_id, reporter="r", case_name="c", status=0,
                            verdict=verdict, cluster_ids=[], normalized_citations=[])


def test_gate_passes_when_all_no_match():
    g = evaluate_gate([_rec(Verdict.NO_MATCH, "a"), _rec(Verdict.NO_MATCH, "b")])
    assert g.state == GateState.PASS
    assert g.n_no_match == 2
    assert g.n_total == 2


def test_gate_fails_when_any_real_match():
    recs = [
        CrosscheckRecord(id="bad", reporter="384 U.S. 436", case_name="Miranda",
                         status=200, verdict=Verdict.REAL_MATCH, cluster_ids=[1],
                         normalized_citations=[]),
        _rec(Verdict.NO_MATCH, "ok"),
    ]
    g = evaluate_gate(recs)
    assert g.state == GateState.FAIL
    assert g.real_match_ids == ["bad"]


def test_gate_incomplete_when_unverifiable_but_no_real_match():
    g = evaluate_gate([_rec(Verdict.NO_MATCH, "a"), _rec(Verdict.UNVERIFIABLE, "b")])
    assert g.state == GateState.INCOMPLETE
    assert g.unverifiable_ids == ["b"]


# ---- run_crosscheck (async, mocked) -------------------------------------------

@pytest.mark.asyncio
async def test_run_crosscheck_requires_token(monkeypatch):
    monkeypatch.delenv("COURTLISTENER_API_TOKEN", raising=False)
    records = [LegalUnanswerable(id="a", reporter="999 U.S. 22187", case_name="Chen")]
    async with httpx.AsyncClient() as client:
        with pytest.raises(CourtListenerTokenMissing):
            await run_crosscheck(records, client, sleep=_noop_sleep)


@pytest.mark.asyncio
@respx.mock
async def test_run_crosscheck_classifies_impossible_citation_as_no_match(monkeypatch):
    monkeypatch.setenv("COURTLISTENER_API_TOKEN", "test-token")
    respx.post(CITATION_LOOKUP_URL).mock(
        return_value=httpx.Response(
            200,
            json=[{"citation": "999 U.S. 22187",
                   "normalized_citations": ["999 U.S. 22187"],
                   "status": 404, "clusters": [], "error_message": "not found"}],
        )
    )
    records = [LegalUnanswerable(id="a", reporter="999 U.S. 22187", case_name="Chen")]
    async with httpx.AsyncClient() as client:
        out = await run_crosscheck(records, client, sleep=_noop_sleep)
    assert out[0].verdict == Verdict.NO_MATCH


@pytest.mark.asyncio
@respx.mock
async def test_run_crosscheck_sends_auth_header(monkeypatch):
    monkeypatch.setenv("COURTLISTENER_API_TOKEN", "secret")
    route = respx.post(CITATION_LOOKUP_URL).mock(
        return_value=httpx.Response(200, json=[
            {"citation": "999 U.S. 22187", "normalized_citations": ["999 U.S. 22187"],
             "status": 404, "clusters": []}])
    )
    records = [LegalUnanswerable(id="a", reporter="999 U.S. 22187", case_name="Chen")]
    async with httpx.AsyncClient() as client:
        await run_crosscheck(records, client, sleep=_noop_sleep)
    assert route.calls.last.request.headers["Authorization"] == "Token secret"


@pytest.mark.asyncio
@respx.mock
async def test_run_crosscheck_retries_on_http_429(monkeypatch):
    monkeypatch.setenv("COURTLISTENER_API_TOKEN", "t")
    respx.post(CITATION_LOOKUP_URL).mock(side_effect=[
        httpx.Response(429, json={"wait_until": "2026-05-26T00:00:00Z"}),
        httpx.Response(200, json=[
            {"citation": "999 U.S. 22187", "normalized_citations": ["999 U.S. 22187"],
             "status": 404, "clusters": []}]),
    ])
    records = [LegalUnanswerable(id="a", reporter="999 U.S. 22187", case_name="Chen")]
    async with httpx.AsyncClient() as client:
        out = await run_crosscheck(records, client, sleep=_noop_sleep)
    assert out[0].verdict == Verdict.NO_MATCH


# ---- write_crosscheck_log -----------------------------------------------------

def test_write_crosscheck_log_emits_one_line_per_record_plus_summary(tmp_path):
    recs = [
        CrosscheckRecord(id="a", reporter="999 U.S. 22187", case_name="Chen",
                         status=404, verdict=Verdict.NO_MATCH, cluster_ids=[],
                         normalized_citations=["999 U.S. 22187"]),
    ]
    gate = evaluate_gate(recs)
    out = tmp_path / "crosscheck.jsonl"
    write_crosscheck_log(recs, gate, out)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    # one record line + one summary line
    assert len(lines) == 2
    rec0 = json.loads(lines[0])
    assert rec0["id"] == "a"
    assert rec0["verdict"] == "no_match"
    summary = json.loads(lines[-1])
    assert summary["type"] == "summary"
    assert summary["gate_state"] == "pass"
