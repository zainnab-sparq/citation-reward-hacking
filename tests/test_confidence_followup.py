"""Contract tests for src/confidence_followup.py (Phase B second-turn confidence).

Per plan §16: second batch turn per generation sends the model its own
prior conversation (original prompt + its response) and asks it to rate
0-100 its confidence per citation. Output keyed identically to
generations/extractions/verifications so aggregation joins 1:1.

The follow-up is a continuation — same generator model continues the
conversation. Messages are [user(original_prompt), assistant(response),
user(confidence-elicit question)]. Forced record_confidences tool use
for structured output.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.elicit import BatchKey, PromptCondition


@pytest.fixture
def conditions() -> list[PromptCondition]:
    return [
        PromptCondition(
            name="citations_expected",
            template="{question}\n\nCite sources.",
        ),
        PromptCondition(name="neutral", template="{question}"),
    ]


@pytest.fixture
def question_text_map() -> dict[str, tuple[str, str]]:
    """{question_id: (text, domain)} so confidence_followup can re-render prompt."""
    return {
        "academic_known_real_0000": (
            "What introduced the Transformer?",
            "academic",
        ),
        "legal_unanswerable_0001": ("Cite a federal case ...", "legal"),
    }


@pytest.fixture
def two_generations() -> list[dict]:
    return [
        {
            "question_id": "academic_known_real_0000",
            "condition": "citations_expected",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "output_text": "Vaswani et al. (2017).",
            "input_tokens": 30,
            "output_tokens": 20,
            "finish_reason": "end_turn",
            "timestamp": "2026-06-04T04:00:00+00:00",
        },
        {
            "question_id": "legal_unanswerable_0001",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 3,
            "output_text": "I cannot identify a real federal case.",
            "input_tokens": 30,
            "output_tokens": 20,
            "finish_reason": "end_turn",
            "timestamp": "2026-06-04T04:00:01+00:00",
        },
    ]


@pytest.fixture
def citations_by_key() -> dict[tuple[str, str, str, int], int]:
    """{BatchKey-tuple: citation_count}. Generation 1 has 2 cites, generation 2 has 0."""
    return {
        (
            "academic_known_real_0000",
            "citations_expected",
            "claude-haiku-4-5-20251001",
            0,
        ): 2,
        (
            "legal_unanswerable_0001",
            "neutral",
            "claude-haiku-4-5-20251001",
            3,
        ): 0,
    }


# ---------- ConfidenceRecord ----------


def test_confidence_record_to_jsonl_dict_roundtrips():
    from src.confidence_followup import ConfidenceRecord

    record = ConfidenceRecord(
        question_id="academic_known_real_0000",
        condition="citations_expected",
        model_id="claude-haiku-4-5-20251001",
        trial_id=0,
        confidences=[
            {"citation_index": 0, "confidence": 95},
            {"citation_index": 1, "confidence": 60},
        ],
        queried_at="2026-06-04T06:00:00+00:00",
    )
    d = record.to_jsonl_dict()
    assert d["trial_id"] == 0
    assert len(d["confidences"]) == 2
    assert d["confidences"][0]["confidence"] == 95


# ---------- build_followup_requests ----------


def test_build_followup_requests_skips_zero_citation_generations(
    two_generations, citations_by_key, conditions, question_text_map
):
    """The legal generation has 0 extracted citations — nothing to rate."""
    from src.confidence_followup import build_followup_requests

    reqs = build_followup_requests(
        generations=two_generations,
        citations_by_key=citations_by_key,
        conditions=conditions,
        question_text_map=question_text_map,
        max_output_tokens=512,
    )
    assert len(reqs) == 1
    key = BatchKey.decode(reqs[0]["custom_id"])
    assert key.question_id == "academic_known_real_0000"


def test_build_followup_requests_messages_have_user_assistant_user_shape(
    two_generations, citations_by_key, conditions, question_text_map
):
    from src.confidence_followup import build_followup_requests

    reqs = build_followup_requests(
        generations=two_generations,
        citations_by_key=citations_by_key,
        conditions=conditions,
        question_text_map=question_text_map,
        max_output_tokens=512,
    )
    messages = reqs[0]["params"]["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["role"] == "user"


def test_build_followup_requests_first_user_message_is_rendered_original_prompt(
    two_generations, citations_by_key, conditions, question_text_map
):
    """The first user message must be exactly the rendered original prompt
    from the (question, condition) pair — otherwise the model is being asked
    to rate confidence in a response to a different prompt."""
    from src.confidence_followup import build_followup_requests

    reqs = build_followup_requests(
        generations=two_generations,
        citations_by_key=citations_by_key,
        conditions=conditions,
        question_text_map=question_text_map,
        max_output_tokens=512,
    )
    expected_prompt = "What introduced the Transformer?\n\nCite sources."
    assert reqs[0]["params"]["messages"][0]["content"] == expected_prompt


def test_build_followup_requests_assistant_message_is_original_output(
    two_generations, citations_by_key, conditions, question_text_map
):
    from src.confidence_followup import build_followup_requests

    reqs = build_followup_requests(
        generations=two_generations,
        citations_by_key=citations_by_key,
        conditions=conditions,
        question_text_map=question_text_map,
        max_output_tokens=512,
    )
    assert reqs[0]["params"]["messages"][1]["content"] == "Vaswani et al. (2017)."


def test_build_followup_requests_third_message_asks_for_per_citation_rating(
    two_generations, citations_by_key, conditions, question_text_map
):
    from src.confidence_followup import build_followup_requests

    reqs = build_followup_requests(
        generations=two_generations,
        citations_by_key=citations_by_key,
        conditions=conditions,
        question_text_map=question_text_map,
        max_output_tokens=512,
    )
    third = reqs[0]["params"]["messages"][2]["content"]
    assert "0" in third and "100" in third
    assert "citation" in third.lower()


def test_build_followup_requests_params_use_generator_model(
    two_generations, citations_by_key, conditions, question_text_map
):
    """The follow-up is a continuation — it must run on the SAME model that
    produced the original answer, not on a separate judge."""
    from src.confidence_followup import build_followup_requests

    reqs = build_followup_requests(
        generations=two_generations,
        citations_by_key=citations_by_key,
        conditions=conditions,
        question_text_map=question_text_map,
        max_output_tokens=512,
    )
    assert reqs[0]["params"]["model"] == "claude-haiku-4-5-20251001"


def test_build_followup_requests_forces_record_confidences_tool(
    two_generations, citations_by_key, conditions, question_text_map
):
    from src.confidence_followup import build_followup_requests

    reqs = build_followup_requests(
        generations=two_generations,
        citations_by_key=citations_by_key,
        conditions=conditions,
        question_text_map=question_text_map,
        max_output_tokens=512,
    )
    tools = reqs[0]["params"]["tools"]
    assert any(t["name"] == "record_confidences" for t in tools)
    assert reqs[0]["params"]["tool_choice"] == {
        "type": "tool",
        "name": "record_confidences",
    }


def test_build_followup_requests_skips_errored_generations(
    citations_by_key, conditions, question_text_map
):
    from src.confidence_followup import build_followup_requests

    gens = [
        {
            "question_id": "academic_known_real_0000",
            "condition": "citations_expected",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "output_text": "",
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": None,
            "timestamp": "2026-06-04T04:00:00+00:00",
            "error": "rate",
        }
    ]
    reqs = build_followup_requests(
        generations=gens,
        citations_by_key=citations_by_key,
        conditions=conditions,
        question_text_map=question_text_map,
        max_output_tokens=512,
    )
    assert reqs == []


def test_build_followup_requests_raises_on_unknown_condition(
    two_generations, citations_by_key, question_text_map
):
    """If a generation references a condition name we don't have a template
    for, we cannot reconstruct the original prompt — surface that loudly."""
    from src.confidence_followup import build_followup_requests
    from src.elicit import PromptCondition

    only_neutral = [PromptCondition(name="neutral", template="{question}")]
    with pytest.raises(KeyError, match="citations_expected"):
        build_followup_requests(
            generations=two_generations,
            citations_by_key=citations_by_key,
            conditions=only_neutral,
            question_text_map=question_text_map,
            max_output_tokens=512,
        )


# ---------- parse_followup_results ----------


def _succeeded_tool_use(custom_id: str, confidences: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="record_confidences",
                        input={"confidences": confidences},
                    ),
                ],
            ),
        ),
    )


def test_parse_followup_results_yields_record_per_succeeded():
    from src.confidence_followup import parse_followup_results

    cid = BatchKey(
        "academic_known_real_0000",
        "claude-haiku-4-5-20251001",
        "citations_expected",
        0,
    ).encode()
    items = [
        _succeeded_tool_use(
            cid,
            [
                {"citation_index": 0, "confidence": 95},
                {"citation_index": 1, "confidence": 60},
            ],
        )
    ]
    records = list(
        parse_followup_results(items, queried_at="2026-06-04T06:00:00+00:00")
    )
    assert len(records) == 1
    assert records[0].confidences[0]["confidence"] == 95
    assert records[0].queried_at == "2026-06-04T06:00:00+00:00"


def test_parse_followup_results_errored_yields_record_with_empty_confidences():
    from src.confidence_followup import parse_followup_results

    cid = BatchKey(
        "academic_known_real_0000",
        "claude-haiku-4-5-20251001",
        "citations_expected",
        0,
    ).encode()
    items = [
        SimpleNamespace(
            custom_id=cid,
            result=SimpleNamespace(
                type="errored",
                error=SimpleNamespace(message="rate"),
            ),
        )
    ]
    records = list(
        parse_followup_results(items, queried_at="2026-06-04T06:00:00+00:00")
    )
    assert len(records) == 1
    assert records[0].confidences == []
    assert records[0].error == "rate"


def test_parse_followup_results_text_only_response_yields_empty_confidences():
    from src.confidence_followup import parse_followup_results

    cid = BatchKey(
        "academic_known_real_0000",
        "claude-haiku-4-5-20251001",
        "citations_expected",
        0,
    ).encode()
    items = [
        SimpleNamespace(
            custom_id=cid,
            result=SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="declined")]
                ),
            ),
        )
    ]
    records = list(
        parse_followup_results(items, queried_at="2026-06-04T06:00:00+00:00")
    )
    assert len(records) == 1
    assert records[0].confidences == []


# ---------- Helper: build citation count map from extractions JSONL ----------


def test_build_citations_by_key_from_extractions_path(tmp_path: Path):
    from src.confidence_followup import build_citations_by_key_from_path

    extractions = [
        {
            "question_id": "academic_known_real_0000",
            "condition": "citations_expected",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "citations": [{"raw_span": "A", "title": "A"}, {"raw_span": "B", "title": "B"}],
        },
        {
            "question_id": "legal_unanswerable_0001",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 3,
            "citations": [],
        },
    ]
    path = tmp_path / "e.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in extractions) + "\n", encoding="utf-8"
    )
    m = build_citations_by_key_from_path(path)
    assert m[
        (
            "academic_known_real_0000",
            "citations_expected",
            "claude-haiku-4-5-20251001",
            0,
        )
    ] == 2
    assert m[
        ("legal_unanswerable_0001", "neutral", "claude-haiku-4-5-20251001", 3)
    ] == 0


# ---------- CLI argparse ----------


def test_confidence_followup_parse_args_defaults_and_required():
    """CLI must accept the symmetric arg set used by the Step 0 dress
    rehearsal shell pipeline. Default max_output_tokens is 512 per plan §16."""
    from src.confidence_followup import parse_args

    args = parse_args(
        [
            "--generations",
            "data/generations/pilot_b/claude-haiku-4-5-20251001.jsonl",
            "--extractions",
            "data/extractions/pilot_b/claude-haiku-4-5-20251001.jsonl",
            "--questions",
            "data/questions/pilot_15_phase_b.jsonl",
            "--output",
            "data/confidences/pilot_b/claude-haiku-4-5-20251001.jsonl",
            "--manifest",
            "data/confidences/pilot_b/batches.manifest.jsonl",
        ]
    )
    assert args.generations == Path(
        "data/generations/pilot_b/claude-haiku-4-5-20251001.jsonl"
    )
    assert args.extractions == Path(
        "data/extractions/pilot_b/claude-haiku-4-5-20251001.jsonl"
    )
    assert args.questions == Path("data/questions/pilot_15_phase_b.jsonl")
    assert args.output == Path(
        "data/confidences/pilot_b/claude-haiku-4-5-20251001.jsonl"
    )
    assert args.manifest == Path("data/confidences/pilot_b/batches.manifest.jsonl")
    assert args.max_output_tokens == 512


def test_confidence_followup_parse_args_overrides_max_tokens():
    from src.confidence_followup import parse_args

    args = parse_args(
        [
            "--generations",
            "g.jsonl",
            "--extractions",
            "e.jsonl",
            "--questions",
            "q.jsonl",
            "--output",
            "c.jsonl",
            "--manifest",
            "m.jsonl",
            "--max-output-tokens",
            "256",
        ]
    )
    assert args.max_output_tokens == 256


def test_confidence_followup_build_question_text_map_from_path(tmp_path: Path):
    """confidence_followup needs (question_id -> (text, domain)). Helper builds
    it from the frozen Phase B JSONL (mapping `question` field -> text)."""
    from src.confidence_followup import build_question_text_map_from_path

    path = tmp_path / "q.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "academic_known_real_0000",
                        "domain": "academic",
                        "question": "What is X?",
                        "expected_existence": "answerable",
                    }
                ),
                json.dumps(
                    {
                        "id": "legal_unanswerable_0001",
                        "domain": "legal",
                        "question": "Cite the fabricated case.",
                        "expected_existence": "unanswerable",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    m = build_question_text_map_from_path(path)
    assert m == {
        "academic_known_real_0000": ("What is X?", "academic"),
        "legal_unanswerable_0001": ("Cite the fabricated case.", "legal"),
    }
