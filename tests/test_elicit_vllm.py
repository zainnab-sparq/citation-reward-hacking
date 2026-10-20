"""Tests for the local-vLLM generation adapter in src/elicit.py (Phase C).

Phase C runs the registered E1/E2 confirmatory sweep on open-weights checkpoints
(Tulu-3 8B/70B base/SFT/DPO/RLVR, OLMo-2-13B base/SFT/DPO/Instruct) via vLLM.

vLLM is NOT installed on the dev box (no GPU), so every vLLM dependency is
injected: the engine (`llm`) and the SamplingParams class are constructor args.
These tests pass fakes; production lazy-imports the real classes. This mirrors
how AnthropicGenerator accepts an optional `client`.

The scientifically load-bearing branch under test: base checkpoints have no
chat template, so they must use the raw-completion path (`llm.generate`), while
post-trained checkpoints (SFT/DPO/RLVR/Instruct) use the chat path
(`llm.chat`). Applying a chat template to a base model would corrupt the
base-vs-post-trained contrast E1 is designed to measure.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import json
from pathlib import Path

from src.elicit import (
    GenerationSpec,
    PromptCondition,
    Question,
    VLLMGenerator,
    _parse_vllm_output,
    _vllm_sampling_kwargs,
    generate_offline_batch,
)


def _fake_request_output(text: str, n_prompt_tokens: int, n_out_tokens: int,
                         finish_reason: str = "stop") -> SimpleNamespace:
    """Mimic vllm.RequestOutput shape closely enough for parsing."""
    return SimpleNamespace(
        prompt_token_ids=list(range(n_prompt_tokens)),
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=list(range(n_out_tokens)),
                finish_reason=finish_reason,
            )
        ],
    )


class _FakeLLM:
    """Records which vLLM entrypoint was called and echoes one output per input."""

    def __init__(self) -> None:
        self.generate_calls: list = []
        self.chat_calls: list = []

    def generate(self, prompts, sampling_params):
        self.generate_calls.append((prompts, sampling_params))
        return [_fake_request_output(f"gen:{p}", 3, 2) for p in prompts]

    def chat(self, messages, sampling_params):
        self.chat_calls.append((messages, sampling_params))
        return [_fake_request_output(f"chat:{m[-1]['content']}", 5, 4) for m in messages]


def _base_spec() -> GenerationSpec:
    return GenerationSpec(
        model_id="allenai/Llama-3.1-Tulu-3-8B",
        provider="vllm_local",
        temperature=0.0,
        max_output_tokens=1024,
    )


# ---------------- _vllm_sampling_kwargs (pure) ----------------

def test_sampling_kwargs_maps_temperature_and_max_tokens():
    kwargs = _vllm_sampling_kwargs(_base_spec())
    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 1024
    assert "seed" not in kwargs


def test_sampling_kwargs_includes_seed_when_in_extra():
    spec = GenerationSpec(
        model_id="m", provider="vllm_local", temperature=0.0,
        max_output_tokens=256, extra={"seed": 20260614},
    )
    kwargs = _vllm_sampling_kwargs(spec)
    assert kwargs["seed"] == 20260614


# ---------------- _parse_vllm_output (pure) ----------------

def test_parse_vllm_output_extracts_text_tokens_and_finish_reason():
    ro = _fake_request_output("Smith (2019) shows...", n_prompt_tokens=7, n_out_tokens=11,
                              finish_reason="length")
    parsed = _parse_vllm_output(ro)
    assert parsed["output_text"] == "Smith (2019) shows..."
    assert parsed["input_tokens"] == 7
    assert parsed["output_tokens"] == 11
    assert parsed["finish_reason"] == "length"


# ---------------- VLLMGenerator.generate_many (batched) ----------------

def test_generate_many_uses_raw_completion_for_base_model():
    llm = _FakeLLM()
    gen = VLLMGenerator(llm=llm, apply_chat_template=False, sampling_params_cls=dict)
    results = gen.generate_many(["P one", "P two"], _base_spec())
    assert llm.generate_calls and not llm.chat_calls
    assert llm.generate_calls[0][0] == ["P one", "P two"]
    assert [r["output_text"] for r in results] == ["gen:P one", "gen:P two"]


def test_generate_many_uses_chat_for_post_trained_model():
    llm = _FakeLLM()
    gen = VLLMGenerator(llm=llm, apply_chat_template=True, sampling_params_cls=dict)
    results = gen.generate_many(["hello"], _base_spec())
    assert llm.chat_calls and not llm.generate_calls
    # chat path wraps each prompt as a single user-turn conversation
    messages = llm.chat_calls[0][0]
    assert messages == [[{"role": "user", "content": "hello"}]]
    assert results[0]["output_text"] == "chat:hello"


def test_generate_many_preserves_input_order_and_parses_all():
    llm = _FakeLLM()
    gen = VLLMGenerator(llm=llm, apply_chat_template=False, sampling_params_cls=dict)
    results = gen.generate_many(["a", "b", "c"], _base_spec())
    assert [r["output_text"] for r in results] == ["gen:a", "gen:b", "gen:c"]
    assert all(set(r) == {"output_text", "input_tokens", "output_tokens", "finish_reason"}
               for r in results)


def test_spec_extra_overrides_chat_template_default():
    llm = _FakeLLM()
    # generator default says chat, but the spec marks this checkpoint as base
    gen = VLLMGenerator(llm=llm, apply_chat_template=True, sampling_params_cls=dict)
    spec = GenerationSpec(
        model_id="allenai/Llama-3.1-Tulu-3-8B", provider="vllm_local",
        temperature=0.0, max_output_tokens=512,
        extra={"apply_chat_template": False},
    )
    gen.generate_many(["x"], spec)
    assert llm.generate_calls and not llm.chat_calls


# ---------------- VLLMGenerator.generate (async, Protocol) ----------------

def test_generate_single_returns_one_result_dict():
    llm = _FakeLLM()
    gen = VLLMGenerator(llm=llm, apply_chat_template=False, sampling_params_cls=dict)
    result = asyncio.run(gen.generate("solo prompt", _base_spec()))
    assert result["output_text"] == "gen:solo prompt"
    assert set(result) == {"output_text", "input_tokens", "output_tokens", "finish_reason"}


# ---------------- generate_offline_batch (sync orchestrator) ----------------

class _FakeBatchGen:
    """Generator exposing only generate_many (the offline-batch contract)."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def generate_many(self, prompts, spec):
        self.calls.append(list(prompts))
        return [
            {"output_text": f"out:{p}", "input_tokens": 4,
             "output_tokens": 6, "finish_reason": "stop"}
            for p in prompts
        ]


def _two_questions() -> list[Question]:
    return [
        Question(id="academic_obscure_real_0001", domain="academic",
                 text="Q-one", expected_existence="answerable"),
        Question(id="medical_unanswerable_0002", domain="medical",
                 text="Q-two", expected_existence="unanswerable"),
    ]


def _two_conditions() -> list[PromptCondition]:
    return [
        PromptCondition(name="neutral", template="{question}"),
        PromptCondition(name="citations_expected", template="Cite. {question}"),
    ]


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_offline_batch_writes_one_record_per_question_condition(tmp_path):
    gen = _FakeBatchGen()
    out = tmp_path / "tulu8b.jsonl"
    generate_offline_batch(_two_questions(), _two_conditions(), _base_spec(), gen, out)
    records = _read_records(out)
    assert len(records) == 4
    keys = {(r["question_id"], r["condition"]) for r in records}
    assert keys == {
        ("academic_obscure_real_0001", "neutral"),
        ("academic_obscure_real_0001", "citations_expected"),
        ("medical_unanswerable_0002", "neutral"),
        ("medical_unanswerable_0002", "citations_expected"),
    }
    one = next(r for r in records if r["condition"] == "neutral"
              and r["question_id"] == "academic_obscure_real_0001")
    assert one["output_text"] == "out:Q-one"
    assert one["model_id"] == "allenai/Llama-3.1-Tulu-3-8B"
    assert one["output_tokens"] == 6


def test_offline_batch_is_single_batched_call(tmp_path):
    gen = _FakeBatchGen()
    generate_offline_batch(_two_questions(), _two_conditions(), _base_spec(),
                           gen, tmp_path / "o.jsonl")
    # all 4 prompts go through vLLM in exactly one continuous-batching call
    assert len(gen.calls) == 1
    assert len(gen.calls[0]) == 4


def test_offline_batch_resumes_and_skips_existing(tmp_path):
    out = tmp_path / "resume.jsonl"
    gen1 = _FakeBatchGen()
    generate_offline_batch(_two_questions()[:1], _two_conditions(), _base_spec(), gen1, out)
    assert len(_read_records(out)) == 2
    # second pass over the full set must only generate the 2 new (q,cond) pairs
    gen2 = _FakeBatchGen()
    generate_offline_batch(_two_questions(), _two_conditions(), _base_spec(), gen2, out)
    assert len(gen2.calls[0]) == 2
    assert len(_read_records(out)) == 4


def test_offline_batch_noop_when_all_present(tmp_path):
    out = tmp_path / "done.jsonl"
    generate_offline_batch(_two_questions(), _two_conditions(), _base_spec(), _FakeBatchGen(), out)
    gen = _FakeBatchGen()
    generate_offline_batch(_two_questions(), _two_conditions(), _base_spec(), gen, out)
    # nothing left to do -> generate_many never invoked
    assert gen.calls == []
