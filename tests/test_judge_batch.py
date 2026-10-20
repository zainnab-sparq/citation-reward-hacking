"""Contract tests for src/judge_batch.py (Phase B citation LLM-judge).

Brought forward from pre-reg §6 Phase H per §11 amendment #5 (Step 0 §8 A.5
gate fail). The judge is Sonnet 4.6 + web_search + a record_citation_judgment
final tool; the input is one batch request per (generation, citation_index);
the output is one judgment per citation.

Key shape adds a `c<digits>` citation_index suffix to the BatchKey form so
custom_id round-trips back to (question_id, model_snapshot, condition,
trial_id, citation_index).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


# ---------- JudgeKey ----------


def test_judge_key_round_trip():
    from src.judge_batch import JudgeKey

    key = JudgeKey(
        question_id="academic-known-real-001",
        model_snapshot="claude-haiku-4-5-20251001",
        condition="neutral",
        trial_id=2,
        citation_index=0,
    )
    encoded = key.encode()
    decoded = JudgeKey.decode(encoded)
    assert decoded == key


def test_judge_key_rejects_malformed_custom_id():
    from src.judge_batch import JudgeKey

    with pytest.raises(ValueError):
        JudgeKey.decode("only__three__parts__t0")  # missing c<idx> suffix


def test_judge_key_citation_index_component_validated():
    from src.judge_batch import JudgeKey

    with pytest.raises(ValueError):
        JudgeKey.decode("q__m__cond__t0__cx")  # citation index not digits


# ---------- JudgmentRecord ----------


def test_judgment_record_to_jsonl_dict_roundtrips():
    from src.judge_batch import JudgmentRecord

    rec = JudgmentRecord(
        question_id="academic-known-real-001",
        condition="neutral",
        model_id="claude-haiku-4-5-20251001",
        trial_id=2,
        citation_index=0,
        judge_model_id="claude-sonnet-4-6",
        judged_at="2026-06-06T12:00:00+00:00",
        judge_status="REAL",
        judge_reasoning="Crossref DOI resolves; authors match.",
        judge_evidence_url="https://doi.org/10.18653/v1/N19-1423",
        judge_search_count=2,
        error=None,
    )
    d = rec.to_jsonl_dict()
    assert d["judge_status"] == "REAL"
    assert d["judge_evidence_url"] == "https://doi.org/10.18653/v1/N19-1423"
    assert d["citation_index"] == 0


# ---------- build_judge_requests ----------


@pytest.fixture
def two_citation_inputs() -> list[dict]:
    return [
        {
            "question_id": "academic-known-real-001",
            "condition": "citations_expected",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "citation_index": 0,
            "domain": "academic",
            "raw_span": "Vaswani et al. (2017). Attention is all you need.",
            "title": "Attention is all you need",
            "case_name": None,
            "authors": ["Vaswani"],
            "year": 2017,
            "venue": "NeurIPS",
            "doi": None,
            "output_excerpt": "...The Transformer was introduced in Vaswani et al. (2017). Attention is all you need. NeurIPS...",
        },
        {
            "question_id": "legal-known-real-002",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 1,
            "citation_index": 1,
            "domain": "legal",
            "raw_span": "Loving v. Virginia, 388 U.S. 1 (1967)",
            "title": None,
            "case_name": "Loving v. Virginia",
            "authors": [],
            "year": 1967,
            "venue": None,
            "doi": None,
            "output_excerpt": "...The Supreme Court struck down anti-miscegenation laws in Loving v. Virginia, 388 U.S. 1 (1967)...",
        },
    ]


def test_build_judge_requests_creates_one_per_citation(two_citation_inputs):
    from src.judge_batch import build_judge_requests

    reqs = build_judge_requests(
        citations=two_citation_inputs,
        judge_model_id="claude-sonnet-4-6",
        max_output_tokens=1024,
        web_search_max_uses=3,
    )
    assert len(reqs) == 2


def test_build_judge_request_includes_web_search_tool(two_citation_inputs):
    from src.judge_batch import build_judge_requests

    reqs = build_judge_requests(
        citations=two_citation_inputs[:1],
        judge_model_id="claude-sonnet-4-6",
        max_output_tokens=1024,
        web_search_max_uses=3,
    )
    tools = reqs[0]["params"]["tools"]
    web_search = next(
        (t for t in tools if t.get("name") == "web_search"), None
    )
    assert web_search is not None, f"tools = {tools!r}"
    # Anthropic's server-tool versioned form
    assert web_search.get("type", "").startswith("web_search_")
    assert web_search.get("max_uses") == 3


def test_build_judge_request_includes_record_citation_judgment_tool(
    two_citation_inputs,
):
    from src.judge_batch import build_judge_requests

    reqs = build_judge_requests(
        citations=two_citation_inputs[:1],
        judge_model_id="claude-sonnet-4-6",
        max_output_tokens=1024,
        web_search_max_uses=3,
    )
    tools = reqs[0]["params"]["tools"]
    rcj = next(
        (t for t in tools if t.get("name") == "record_citation_judgment"),
        None,
    )
    assert rcj is not None
    schema = rcj["input_schema"]
    assert schema["type"] == "object"
    assert "status" in schema["properties"]
    assert schema["properties"]["status"]["enum"] == [
        "REAL",
        "FABRICATED",
        "UNVERIFIABLE",
    ]
    assert "reasoning" in schema["properties"]
    assert "evidence_url" in schema["properties"]
    assert "status" in schema["required"]
    assert "reasoning" in schema["required"]


def test_build_judge_request_uses_tool_choice_auto(two_citation_inputs):
    """tool_choice MUST be 'auto' so web_search and record_citation_judgment
    can both be chosen; forcing record_citation_judgment would prevent the
    judge from doing any web search at all.
    """
    from src.judge_batch import build_judge_requests

    reqs = build_judge_requests(
        citations=two_citation_inputs[:1],
        judge_model_id="claude-sonnet-4-6",
        max_output_tokens=1024,
        web_search_max_uses=3,
    )
    assert reqs[0]["params"]["tool_choice"] == {"type": "auto"}


def test_build_judge_request_prompt_includes_citation_fields(two_citation_inputs):
    from src.judge_batch import build_judge_requests

    reqs = build_judge_requests(
        citations=two_citation_inputs[:1],
        judge_model_id="claude-sonnet-4-6",
        max_output_tokens=1024,
        web_search_max_uses=3,
    )
    messages = reqs[0]["params"]["messages"]
    assert messages[0]["role"] == "user"
    user_text = messages[0]["content"]
    assert "Attention is all you need" in user_text
    assert "Vaswani" in user_text
    assert "2017" in user_text
    assert "NeurIPS" in user_text
    assert "academic" in user_text  # domain
    assert "Transformer was introduced" in user_text  # output_excerpt
    assert "record_citation_judgment" in user_text  # explicit final-action instruction


def test_build_judge_request_legal_prompt_uses_case_name(two_citation_inputs):
    from src.judge_batch import build_judge_requests

    reqs = build_judge_requests(
        citations=[two_citation_inputs[1]],  # the legal one
        judge_model_id="claude-sonnet-4-6",
        max_output_tokens=1024,
        web_search_max_uses=3,
    )
    user_text = reqs[0]["params"]["messages"][0]["content"]
    assert "Loving v. Virginia" in user_text
    assert "1967" in user_text
    assert "legal" in user_text


def test_build_judge_request_custom_id_encodes_citation_index(two_citation_inputs):
    from src.judge_batch import JudgeKey, build_judge_requests

    reqs = build_judge_requests(
        citations=two_citation_inputs,
        judge_model_id="claude-sonnet-4-6",
        max_output_tokens=1024,
        web_search_max_uses=3,
    )
    key0 = JudgeKey.decode(reqs[0]["custom_id"])
    assert key0.question_id == "academic-known-real-001"
    assert key0.citation_index == 0
    key1 = JudgeKey.decode(reqs[1]["custom_id"])
    assert key1.citation_index == 1


# ---------- parse_judgment_results ----------


def _succeeded_item(custom_id: str, tool_input: dict, search_count: int = 0):
    """Build a fake batch result item with a record_citation_judgment tool_use block."""
    tool_use_block = SimpleNamespace(
        type="tool_use",
        name="record_citation_judgment",
        input=tool_input,
    )
    server_tool_blocks = [
        SimpleNamespace(type="server_tool_use", name="web_search")
        for _ in range(search_count)
    ]
    content = server_tool_blocks + [tool_use_block]
    message = SimpleNamespace(content=content)
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=message),
    )


def _errored_item(custom_id: str, message: str = "rate_limit_error"):
    err = SimpleNamespace(message=message, type=message)
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="errored", error=err),
    )


def _succeeded_no_tool_use(custom_id: str, text: str = "I refuse"):
    text_block = SimpleNamespace(type="text", text=text)
    message = SimpleNamespace(content=[text_block])
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="succeeded", message=message),
    )


def test_parse_judgment_extracts_status_and_reasoning():
    from src.judge_batch import JudgeKey, parse_judgment_results

    key = JudgeKey(
        question_id="academic-known-real-001",
        model_snapshot="claude-haiku-4-5-20251001",
        condition="neutral",
        trial_id=0,
        citation_index=0,
    ).encode()
    item = _succeeded_item(
        key,
        {
            "status": "REAL",
            "reasoning": "Crossref DOI resolves; Vaswani et al. NeurIPS 2017.",
            "evidence_url": "https://doi.org/10.48550/arXiv.1706.03762",
        },
        search_count=2,
    )
    records = list(
        parse_judgment_results(
            [item], judge_model_id="claude-sonnet-4-6", judged_at="2026-06-06T12:00:00+00:00"
        )
    )
    assert len(records) == 1
    r = records[0]
    assert r.judge_status == "REAL"
    assert "Crossref" in r.judge_reasoning
    assert r.judge_evidence_url == "https://doi.org/10.48550/arXiv.1706.03762"
    assert r.judge_search_count == 2
    assert r.error is None
    assert r.citation_index == 0


def test_parse_judgment_handles_missing_evidence_url():
    from src.judge_batch import JudgeKey, parse_judgment_results

    key = JudgeKey("q", "m", "neutral", 0, 0).encode()
    item = _succeeded_item(
        key,
        {"status": "UNVERIFIABLE", "reasoning": "Search returned no definitive evidence."},
    )
    records = list(
        parse_judgment_results(
            [item], judge_model_id="claude-sonnet-4-6", judged_at="2026-06-06T12:00:00+00:00"
        )
    )
    assert records[0].judge_evidence_url is None
    assert records[0].judge_status == "UNVERIFIABLE"


def test_parse_judgment_handles_succeeded_without_tool_use_as_error():
    from src.judge_batch import JudgeKey, parse_judgment_results

    key = JudgeKey("q", "m", "neutral", 0, 0).encode()
    item = _succeeded_no_tool_use(key, text="I cannot help with that.")
    records = list(
        parse_judgment_results(
            [item], judge_model_id="claude-sonnet-4-6", judged_at="2026-06-06T12:00:00+00:00"
        )
    )
    r = records[0]
    assert r.judge_status == "UNVERIFIABLE"
    assert r.error is not None
    assert "no_tool_use" in r.error or "missing_judgment" in r.error


def test_parse_judgment_handles_errored_result():
    from src.judge_batch import JudgeKey, parse_judgment_results

    key = JudgeKey("q", "m", "neutral", 0, 0).encode()
    item = _errored_item(key, "rate_limit_error")
    records = list(
        parse_judgment_results(
            [item], judge_model_id="claude-sonnet-4-6", judged_at="2026-06-06T12:00:00+00:00"
        )
    )
    r = records[0]
    assert r.judge_status == "UNVERIFIABLE"
    assert "rate_limit_error" in r.error


def test_parse_judgment_invalid_status_coerced_to_unverifiable():
    """Model emits an out-of-enum status -> coerce to UNVERIFIABLE + error.

    Forced-enum is enforced by the API but a defensive parse keeps the
    aggregator from crashing on a malformed downstream record.
    """
    from src.judge_batch import JudgeKey, parse_judgment_results

    key = JudgeKey("q", "m", "neutral", 0, 0).encode()
    item = _succeeded_item(key, {"status": "MAYBE", "reasoning": "x"})
    records = list(
        parse_judgment_results(
            [item], judge_model_id="claude-sonnet-4-6", judged_at="2026-06-06T12:00:00+00:00"
        )
    )
    r = records[0]
    assert r.judge_status == "UNVERIFIABLE"
    assert "invalid_status" in r.error


# ---------- CLI ----------


def test_parse_args_defaults(tmp_path):
    from src.judge_batch import parse_args

    args = parse_args(
        [
            "--citations",
            str(tmp_path / "in.jsonl"),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.jsonl"),
        ]
    )
    assert args.judge_model == "claude-sonnet-4-6"
    assert args.max_output_tokens == 1024
    assert args.web_search_max_uses == 3
