"""Tests for the Phase C open-weights sweep orchestrator (src/run_phase_c.py).

GPU-free: the vLLM engine is never constructed here. The checkpoint->spec and
checkpoint->engine-kwargs projections are pure, and the sweep loop takes an
injected generator factory so it can run against a fake batched generator.

The load-bearing invariant under test: a base checkpoint (stage_ordinal 0) must
have is_post_trained=0 AND apply_chat_template=false. Validation rejects any
drift, because feeding a base model a chat template would corrupt the
base-vs-post-trained contrast E1 measures.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.elicit import PromptCondition, Question
from src.run_phase_c import (
    CHECKPOINTS_PATH,
    _engine_kwargs_from_row,
    _safe_filename,
    _spec_from_checkpoint_row,
    run_one,
    run_sweep,
    validate_checkpoints,
)
from src.run_phase_b import load_model_registry


def _base_row() -> dict:
    return {
        "model_id": "meta-llama/Llama-3.1-8B", "model_family": "tulu3_8b",
        "vendor": "meta", "scale": "8B", "stage": "base", "stage_ordinal": 0,
        "is_post_trained": 0, "is_open_weights": 1,
        "hf_repo": "meta-llama/Llama-3.1-8B",
        "revision": "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b",
        "apply_chat_template": False, "tensor_parallel_size": 1,
    }


def _post_row() -> dict:
    return {
        "model_id": "allenai/Llama-3.1-Tulu-3-8B", "model_family": "tulu3_8b",
        "vendor": "allenai", "scale": "8B", "stage": "RLVR", "stage_ordinal": 3,
        "is_post_trained": 1, "is_open_weights": 1,
        "hf_repo": "allenai/Llama-3.1-Tulu-3-8B",
        "revision": "666943798adbde0b1aff34626007e26986a3c107",
        "apply_chat_template": True, "tensor_parallel_size": 1,
    }


# ---------------- spec / engine-kwargs projections ----------------

def test_spec_from_checkpoint_row_is_vllm_local_temp0():
    spec = _spec_from_checkpoint_row(_base_row(), seed=20260614)
    assert spec.model_id == "meta-llama/Llama-3.1-8B"
    assert spec.provider == "vllm_local"
    assert spec.temperature == 0.0
    assert spec.extra["seed"] == 20260614
    assert spec.extra["apply_chat_template"] is False
    assert spec.extra["revision"] == "d04e592bb4f6aa9cfee91e2e20afa771667e1d4b"


def test_engine_kwargs_pins_revision_tp_and_full_precision():
    kwargs = _engine_kwargs_from_row(_post_row())
    assert kwargs["revision"] == "666943798adbde0b1aff34626007e26986a3c107"
    assert kwargs["tensor_parallel_size"] == 1
    # full precision: quantization can shift hallucination rates (plan section 5)
    assert kwargs["dtype"] == "bfloat16"
    # ample for our prompts (<=~1500 tok); neutral at temp=0, relieves KV pressure
    assert kwargs["max_model_len"] == 4096


def test_engine_kwargs_single_gpu_omits_gpu_mem_util():
    # TP=1 (8B/13B on one 80GB card) has ample headroom -> use vLLM default
    assert "gpu_memory_utilization" not in _engine_kwargs_from_row(_post_row())


def test_engine_kwargs_multi_gpu_sets_gpu_mem_util():
    row = _post_row()
    row["tensor_parallel_size"] = 2
    kwargs = _engine_kwargs_from_row(row)
    assert kwargs["tensor_parallel_size"] == 2
    # KV headroom for 70B on 2x80GB (weights ~70GB/GPU)
    assert kwargs["gpu_memory_utilization"] == 0.96


# ---------------- validate_checkpoints (invariants) ----------------

def test_validate_passes_on_committed_registry():
    rows = load_model_registry(CHECKPOINTS_PATH)
    validated = validate_checkpoints(rows)
    # 12 from the three Phase C cascades, plus the three registered breadth
    # checkpoints collected 2026-09-06 under the pre-committed rule in
    # docs/sub_amendment_13_roster_reduction_draft.md (Llama-3.1-70B-Instruct
    # and the Qwen2.5-72B base/Instruct pair).
    assert len(validated) == 15
    # 3 cascade bases + the Qwen2.5-72B base. Llama-3.1-70B-Instruct's base is
    # meta-llama/Llama-3.1-70B, already present as tulu3_70b stage 0.
    assert sum(1 for r in validated if r["stage_ordinal"] == 0) == 4


def test_validate_rejects_non_hex_sha():
    bad = _base_row()
    bad["revision"] = "main"
    with pytest.raises(ValueError, match="revision"):
        validate_checkpoints([bad])


def test_validate_rejects_base_with_chat_template():
    bad = _base_row()
    bad["apply_chat_template"] = True
    with pytest.raises(ValueError, match="base"):
        validate_checkpoints([bad])


def test_validate_rejects_base_marked_post_trained():
    bad = _base_row()
    bad["is_post_trained"] = 1
    with pytest.raises(ValueError, match="post_trained|base"):
        validate_checkpoints([bad])


def test_validate_rejects_duplicate_model_ids():
    with pytest.raises(ValueError, match="duplicate"):
        validate_checkpoints([_base_row(), _base_row()])


# ---------------- run_one (single checkpoint, in-process) ----------------

class _FakeBatchGen:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def generate_many(self, prompts, spec):
        self.calls.append(list(prompts))
        return [
            {"output_text": f"out:{p}", "input_tokens": 4,
             "output_tokens": 6, "finish_reason": "stop"}
            for p in prompts
        ]


def _one_question() -> list[Question]:
    return [Question(id="academic_obscure_real_0001", domain="academic",
                     text="Qx", expected_existence="answerable")]


def _two_conditions() -> list[PromptCondition]:
    return [PromptCondition(name="neutral", template="{question}"),
            PromptCondition(name="citations_expected", template="Cite. {question}")]


def test_run_one_writes_file_for_one_checkpoint(tmp_path):
    built: list[dict] = []

    def factory(row):
        built.append(row)
        return _FakeBatchGen()

    run_one(_base_row(), _one_question(), _two_conditions(), tmp_path,
            seed=7, generator_factory=factory)

    assert len(built) == 1
    f = tmp_path / "meta-llama__Llama-3.1-8B.jsonl"
    recs = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(recs) == 2
    assert {r["condition"] for r in recs} == {"neutral", "citations_expected"}
    assert all(r["model_id"] == "meta-llama/Llama-3.1-8B" for r in recs)


def test_run_one_skips_when_already_complete(tmp_path):
    q, conds = _one_question(), _two_conditions()
    f = tmp_path / "meta-llama__Llama-3.1-8B.jsonl"
    f.write_text("\n".join(
        json.dumps({"question_id": q[0].id, "condition": c.name,
                    "model_id": "meta-llama/Llama-3.1-8B", "output_text": "x"})
        for c in conds) + "\n", encoding="utf-8")
    called: list[dict] = []

    def factory(row):
        called.append(row)
        return _FakeBatchGen()

    run_one(_base_row(), q, conds, tmp_path, seed=7, generator_factory=factory)
    # already complete -> generator never built -> no (expensive) model load
    assert called == []


# ---------------- run_sweep (subprocess isolation per checkpoint) ----------------

def test_run_sweep_spawns_once_per_incomplete_checkpoint(tmp_path):
    rows = [_base_row(), _post_row()]
    expected = 2  # 1 question x 2 conditions
    spawned: list[str] = []

    def fake_spawn(row):
        spawned.append(row["model_id"])
        # emulate the subprocess producing the output file
        (tmp_path / f"{_safe_filename(row['model_id'])}.jsonl").write_text(
            "x\n" * expected, encoding="utf-8")

    run_sweep(rows, tmp_path, expected_per_model=expected, spawn=fake_spawn)
    assert spawned == ["meta-llama/Llama-3.1-8B", "allenai/Llama-3.1-Tulu-3-8B"]


def test_run_sweep_skips_complete_checkpoint(tmp_path):
    rows = [_base_row(), _post_row()]
    expected = 2
    # base already complete on disk -> must not be spawned again
    (tmp_path / "meta-llama__Llama-3.1-8B.jsonl").write_text("x\n" * expected, encoding="utf-8")
    spawned: list[str] = []
    run_sweep(rows, tmp_path, expected_per_model=expected,
              spawn=lambda r: spawned.append(r["model_id"]))
    assert spawned == ["allenai/Llama-3.1-Tulu-3-8B"]
