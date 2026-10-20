"""Tests for elicit.py — model-agnostic citation-elicitation runner."""
import json
from pathlib import Path

import pytest

from src.elicit import (
    AnthropicGenerator,
    GenerationRecord,
    GenerationSpec,
    PromptCondition,
    Question,
    generate_batch,
)


class FakeGenerator:
    """Echos prompt back as output. Used to test orchestration without real API calls."""

    def __init__(self):
        self.calls: list[tuple[str, GenerationSpec]] = []

    async def generate(self, prompt: str, spec: GenerationSpec) -> dict:
        self.calls.append((prompt, spec))
        return {
            "output_text": f"FAKE-RESPONSE: {prompt[:50]}",
            "input_tokens": 10,
            "output_tokens": 5,
            "finish_reason": "end_turn",
        }


@pytest.fixture
def question_neural_attention():
    return Question(
        id="q-001",
        domain="academic",
        text="What paper introduced the Transformer architecture?",
        expected_existence="answerable",
    )


@pytest.fixture
def conditions():
    return [
        PromptCondition(
            name="citations_expected",
            template="{question}\n\nPlease answer with at least two academic citations.",
        ),
        PromptCondition(name="neutral", template="{question}"),
        PromptCondition(
            name="abstention_permitted",
            template=(
                "{question}\n\nIf you are not certain a specific source exists, "
                "say so rather than guessing."
            ),
        ),
    ]


@pytest.fixture
def fake_spec():
    return GenerationSpec(
        model_id="fake-model",
        provider="fake",
        temperature=1.0,
        max_output_tokens=256,
    )


@pytest.mark.asyncio
async def test_generate_batch_produces_one_record_per_question_condition(
    tmp_path: Path, question_neural_attention, conditions, fake_spec
):
    out_path = tmp_path / "generations.jsonl"
    generator = FakeGenerator()

    await generate_batch(
        questions=[question_neural_attention],
        conditions=conditions,
        spec=fake_spec,
        generator=generator,
        output_path=out_path,
    )

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3  # one record per condition
    records = [json.loads(line) for line in lines]
    condition_names = {r["condition"] for r in records}
    assert condition_names == {"citations_expected", "neutral", "abstention_permitted"}
    assert all(r["model_id"] == "fake-model" for r in records)
    assert all("FAKE-RESPONSE" in r["output_text"] for r in records)


@pytest.mark.asyncio
async def test_generate_batch_resumes_skipping_existing_records(
    tmp_path: Path, question_neural_attention, conditions, fake_spec
):
    out_path = tmp_path / "generations.jsonl"

    # Pre-populate with a record for one condition
    pre_existing = GenerationRecord(
        question_id="q-001",
        condition="neutral",
        model_id="fake-model",
        output_text="PRE-EXISTING",
        input_tokens=1,
        output_tokens=1,
        finish_reason="end_turn",
        timestamp="2026-01-01T00:00:00Z",
    )
    out_path.write_text(json.dumps(pre_existing.to_jsonl_dict()) + "\n", encoding="utf-8")

    generator = FakeGenerator()
    await generate_batch(
        questions=[question_neural_attention],
        conditions=conditions,
        spec=fake_spec,
        generator=generator,
        output_path=out_path,
    )

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    # Generator should have been called only twice (the two missing conditions)
    assert len(generator.calls) == 2
    # The pre-existing record should still be there with original text
    records = [json.loads(line) for line in lines]
    neutral_record = next(r for r in records if r["condition"] == "neutral")
    assert neutral_record["output_text"] == "PRE-EXISTING"


@pytest.mark.asyncio
async def test_anthropic_generator_calls_messages_api(monkeypatch):
    """AnthropicGenerator should call the Messages API with the right payload."""
    captured = {}

    class FakeAnthropicResponse:
        def __init__(self):
            self.content = [type("Block", (), {"type": "text", "text": "Test response."})()]
            self.usage = type("Usage", (), {"input_tokens": 12, "output_tokens": 3})()
            self.stop_reason = "end_turn"

    class FakeAnthropicMessages:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return FakeAnthropicResponse()

    class FakeAnthropicClient:
        def __init__(self):
            self.messages = FakeAnthropicMessages()

    gen = AnthropicGenerator(client=FakeAnthropicClient())
    spec = GenerationSpec(
        model_id="claude-sonnet-4-6",
        provider="anthropic",
        temperature=0.7,
        max_output_tokens=128,
    )

    result = await gen.generate("What is gravity?", spec)

    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["max_tokens"] == 128
    assert captured["temperature"] == 0.7
    assert captured["messages"] == [{"role": "user", "content": "What is gravity?"}]
    assert result["output_text"] == "Test response."
    assert result["input_tokens"] == 12
    assert result["output_tokens"] == 3
    assert result["finish_reason"] == "end_turn"
