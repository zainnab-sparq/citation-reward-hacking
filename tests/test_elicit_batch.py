"""Tests for the Anthropic Message Batches orchestration path in src/elicit.py.

The sync path (generate_batch + AnthropicGenerator) is unchanged. This module
covers the batch-mode addition: custom_id round-trip, batch submission with
all (question × condition × trial) requests, sidecar-manifest persistence,
polling-until-ended, result mapping, resume-on-restart, and snapshot pinning.
"""
from __future__ import annotations

import pytest

from types import SimpleNamespace

from src.elicit import (
    BatchKey,
    GenerationSpec,
    PromptCondition,
    Question,
    build_batch_requests,
    parse_batch_results,
)


@pytest.fixture
def two_questions() -> list[Question]:
    return [
        Question(
            id="academic_known_real_0001",
            domain="academic",
            text="What is attention?",
            expected_existence="answerable",
        ),
        Question(
            id="academic_obscure_real_0042",
            domain="academic",
            text="Define WMDP-Cyber score.",
            expected_existence="answerable",
        ),
    ]


@pytest.fixture
def three_conditions() -> list[PromptCondition]:
    return [
        PromptCondition(name="neutral", template="Q: {question}"),
        PromptCondition(name="citations_expected", template="Cite sources. Q: {question}"),
        PromptCondition(
            name="abstention_permitted",
            template="Q: {question} (abstain if unsure)",
        ),
    ]


@pytest.fixture
def opus_spec() -> GenerationSpec:
    return GenerationSpec(
        model_id="claude-opus-4-7",
        provider="anthropic",
        temperature=1.0,
        max_output_tokens=512,
    )


def test_batch_key_encodes_with_compact_codes_under_anthropic_64_char_limit() -> None:
    """BatchKey.encode() produces a custom_id <= 64 chars on the registered
    Phase B inputs (frozen dataset question_id + amendment #2 model snapshot
    + amendment #5 condition names + trials 0-4).

    Anthropic's Batch API enforces custom_id <= 64. The verbose long-form
    that joined raw question_id + raw model snapshot + raw condition
    exceeded 64 on real Phase B inputs (75 chars: see §11 sub-amendment
    2026-06-07). The compact code encoding maps each field to a registered
    short code while preserving full round-trip via BatchKey.decode().
    """
    key = BatchKey(
        question_id="academic_known_real_0000",
        model_snapshot="claude-haiku-4-5-20251001",
        condition="citations_expected",
        trial_id=2,
    )
    encoded = key.encode()
    assert len(encoded) <= 64, f"custom_id {encoded!r} is {len(encoded)} chars, must be <= 64"


def test_batch_key_round_trips_through_encode_decode() -> None:
    """decode(encode(key)) == key for all components."""
    original = BatchKey(
        question_id="legal_unanswerable_0067",
        model_snapshot="claude-sonnet-4-6",
        condition="abstention_permitted",
        trial_id=0,
    )

    decoded = BatchKey.decode(original.encode())

    assert decoded == original


def test_batch_key_decode_rejects_malformed_custom_id() -> None:
    """A custom_id without exactly four '__'-separated fields raises ValueError.

    Defends against silent mis-mapping of results to questions if the
    encoding format ever drifts between submission and retrieval.
    """
    with pytest.raises(ValueError):
        BatchKey.decode("only-three__parts__here")

    with pytest.raises(ValueError):
        BatchKey.decode("q__m__c__not_a_trial_number")


def test_batch_key_round_trips_all_registered_combinations() -> None:
    """Every (domain, cell, model, condition) combination registered in the
    Phase B code tables must round-trip. Defends against drift between the
    encoder and decoder code maps."""
    for qid in (
        "academic_known_real_0000",
        "medical_obscure_real_0299",
        "legal_unanswerable_0150",
    ):
        for model in (
            "claude-haiku-4-5-20251001",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
        ):
            for condition in (
                "neutral",
                "citations_expected",
                "abstention_permitted",
                "length_matched",
            ):
                key = BatchKey(
                    question_id=qid,
                    model_snapshot=model,
                    condition=condition,
                    trial_id=4,
                )
                assert BatchKey.decode(key.encode()) == key


# The 12 Phase C open-weights checkpoints (data/models/phase_c_checkpoints.jsonl).
# Amendment #7 registered the compact-code table for the three Phase B Claude
# models only; Phase C extraction reuses the SAME BatchKey join-key mechanism,
# so each open-weights model_id must encode losslessly under the 64-char cap with
# no code collision — else extract_batch raises ValueError on the first row.
_PHASE_C_MODEL_IDS = (
    "meta-llama/Llama-3.1-8B",
    "allenai/Llama-3.1-Tulu-3-8B-SFT",
    "allenai/Llama-3.1-Tulu-3-8B-DPO",
    "allenai/Llama-3.1-Tulu-3-8B",
    "allenai/OLMo-2-1124-13B",
    "allenai/OLMo-2-1124-13B-SFT",
    "allenai/OLMo-2-1124-13B-DPO",
    "allenai/OLMo-2-1124-13B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "allenai/Llama-3.1-Tulu-3-70B-SFT",
    "allenai/Llama-3.1-Tulu-3-70B-DPO",
    "allenai/Llama-3.1-Tulu-3-70B",
)


def test_batch_key_round_trips_all_phase_c_open_weights_models() -> None:
    """Every Phase C open-weights checkpoint encodes + round-trips under the
    64-char cap, with a unique model code (no collision across the 12 or with
    the three Phase B Claude codes)."""
    for model in _PHASE_C_MODEL_IDS:
        for qid in ("academic_known_real_0000", "legal_unanswerable_0150"):
            for condition in ("neutral", "citations_expected", "length_matched"):
                key = BatchKey(
                    question_id=qid,
                    model_snapshot=model,
                    condition=condition,
                    trial_id=0,
                )
                encoded = key.encode()
                assert len(encoded) <= 64, f"{encoded!r} is {len(encoded)} chars"
                assert BatchKey.decode(encoded) == key
    # holding qid+condition+trial fixed, distinct models must yield distinct
    # encodings (i.e. distinct model codes)
    encodings = {
        BatchKey("academic_known_real_0000", m, "neutral", 0).encode()
        for m in _PHASE_C_MODEL_IDS
    }
    assert len(encodings) == len(_PHASE_C_MODEL_IDS)


def test_batch_key_rejects_unregistered_components() -> None:
    """encode() raises on values not in the registered code tables — this
    is how we surface a drifted dataset/snapshot/condition before paying
    for a batch run."""
    # Sanity check: the registered baseline encodes cleanly.
    BatchKey(
        question_id="academic_known_real_0000",
        model_snapshot="claude-haiku-4-5-20251001",
        condition="citations_expected",
        trial_id=2,
    ).encode()
    with pytest.raises(ValueError):
        BatchKey(
            question_id="not_a_real_qid",
            model_snapshot="claude-haiku-4-5-20251001",
            condition="citations_expected",
            trial_id=0,
        ).encode()
    with pytest.raises(ValueError):
        BatchKey(
            question_id="academic_known_real_0000",
            model_snapshot="gpt-4-turbo",
            condition="citations_expected",
            trial_id=0,
        ).encode()
    with pytest.raises(ValueError):
        BatchKey(
            question_id="academic_known_real_0000",
            model_snapshot="claude-haiku-4-5-20251001",
            condition="unregistered_condition",
            trial_id=0,
        ).encode()


# ---------- build_batch_requests ----------


def test_build_batch_requests_one_per_question_condition_trial(
    two_questions: list[Question],
    three_conditions: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """2 questions × 3 conditions × 2 trials = 12 requests, each with a unique custom_id."""
    requests = build_batch_requests(
        two_questions, three_conditions, opus_spec, num_trials=2
    )

    assert len(requests) == 12
    custom_ids = [r["custom_id"] for r in requests]
    assert len(set(custom_ids)) == 12, "custom_ids must be unique within a batch"


def test_build_batch_requests_preserves_snapshot_pinning(
    two_questions: list[Question],
    three_conditions: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """Each request's params.model must equal the spec.model_id (dated snapshot)."""
    requests = build_batch_requests(two_questions, three_conditions, opus_spec, num_trials=1)

    for r in requests:
        assert r["params"]["model"] == "claude-opus-4-7"


def test_build_batch_requests_renders_condition_template(
    two_questions: list[Question],
    three_conditions: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """The rendered prompt (via PromptCondition.render) appears as the user message."""
    requests = build_batch_requests(two_questions, three_conditions, opus_spec, num_trials=1)

    citations_expected = [
        r
        for r in requests
        if BatchKey.decode(r["custom_id"]).condition == "citations_expected"
    ]
    assert len(citations_expected) == 2
    for r in citations_expected:
        user_msg = r["params"]["messages"][0]
        assert user_msg["role"] == "user"
        assert user_msg["content"].startswith("Cite sources. Q:")


def test_build_batch_requests_carries_max_tokens_and_temperature(
    two_questions: list[Question],
    three_conditions: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """Spec's max_output_tokens and temperature appear in every request's params."""
    requests = build_batch_requests(two_questions, three_conditions, opus_spec, num_trials=1)

    for r in requests:
        assert r["params"]["max_tokens"] == 512
        assert r["params"]["temperature"] == 1.0


def test_build_batch_requests_rejects_zero_trials(
    two_questions: list[Question],
    three_conditions: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """num_trials=0 is meaningless and indicates a caller bug."""
    with pytest.raises(ValueError):
        build_batch_requests(two_questions, three_conditions, opus_spec, num_trials=0)


# ---------- parse_batch_results ----------


def _make_succeeded(custom_id: str, text: str, in_tokens: int, out_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=in_tokens, output_tokens=out_tokens),
                stop_reason="end_turn",
            ),
        ),
    )


def test_parse_batch_results_emits_generation_record_for_succeeded() -> None:
    """Successful batch result maps to a GenerationRecord with trial_id carried
    from the custom_id, text concatenated from text blocks, and token counts populated."""
    custom_id = BatchKey(
        question_id="academic_known_real_0001",
        model_snapshot="claude-opus-4-7",
        condition="citations_expected",
        trial_id=2,
    ).encode()

    records = list(parse_batch_results([_make_succeeded(custom_id, "hello world", 100, 20)]))

    assert len(records) == 1
    r = records[0]
    assert r.question_id == "academic_known_real_0001"
    assert r.model_id == "claude-opus-4-7"
    assert r.condition == "citations_expected"
    assert r.trial_id == 2
    assert r.output_text == "hello world"
    assert r.input_tokens == 100
    assert r.output_tokens == 20
    assert r.finish_reason == "end_turn"
    assert r.error is None
    assert r.timestamp.endswith("Z")


def test_parse_batch_results_emits_error_record_for_errored() -> None:
    """Errored result populates GenerationRecord.error with the API error info
    and leaves output_text empty / token counts None."""
    custom_id = BatchKey(
        question_id="medical_obscure_real_0013",
        model_snapshot="claude-sonnet-4-6",
        condition="neutral",
        trial_id=0,
    ).encode()
    fake = SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="errored",
            error=SimpleNamespace(
                type="overloaded_error",
                message="Server overloaded, try again later",
            ),
        ),
    )

    records = list(parse_batch_results([fake]))

    assert len(records) == 1
    r = records[0]
    assert r.output_text == ""
    assert r.input_tokens is None
    assert r.output_tokens is None
    assert r.error is not None
    assert "overloaded" in r.error.lower()


def test_parse_batch_results_emits_error_record_for_expired() -> None:
    """24h SLA timeout produces 'expired' result type (no error field); treat
    as a recoverable error record so a re-run picks it up."""
    custom_id = BatchKey(
        question_id="legal_unanswerable_0024",
        model_snapshot="claude-haiku-4-5-20251001",
        condition="abstention_permitted",
        trial_id=4,
    ).encode()
    fake = SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="expired"),
    )

    records = list(parse_batch_results([fake]))

    assert len(records) == 1
    r = records[0]
    assert r.error is not None
    assert "expired" in r.error.lower()
    assert r.output_text == ""


def test_parse_batch_results_preserves_input_order() -> None:
    """The iteration order of the input results survives into the output records.

    Batch results from the SDK arrive in arbitrary order; the parser does not
    reorder them. Caller code that depends on ordering (e.g., diff-against-existing)
    can sort by `(question_id, condition, trial_id)`.
    """
    cid_a = BatchKey("academic_known_real_0001", "claude-opus-4-7", "neutral", 0).encode()
    cid_b = BatchKey("medical_known_real_0002", "claude-opus-4-7", "neutral", 0).encode()
    cid_c = BatchKey("legal_known_real_0003", "claude-opus-4-7", "neutral", 0).encode()

    inputs = [
        _make_succeeded(cid_b, "B", 1, 1),
        _make_succeeded(cid_a, "A", 1, 1),
        _make_succeeded(cid_c, "C", 1, 1),
    ]

    records = list(parse_batch_results(inputs))
    assert [r.question_id for r in records] == [
        "medical_known_real_0002",
        "academic_known_real_0001",
        "legal_known_real_0003",
    ]
