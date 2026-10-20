"""Phase C CLI orchestrator: registered E1+E2 confirmatory sweep on open weights.

Glue layer over `generate_offline_batch` + `VLLMGenerator` from src/elicit.py:
  1. Load the same frozen 900-question JSONL and the same registered four
     prompt conditions (src.pilot.CONDITIONS) used in Phase B, so the
     open-weights run is directly comparable to the Claude run.
  2. Load + validate the pinned checkpoint registry
     (data/models/phase_c_checkpoints.jsonl): Tulu-3 8B/70B and OLMo-2-13B at
     base/SFT/DPO/final, each pinned to a HuggingFace commit SHA.
  3. For each checkpoint, run a fresh subprocess (--single-model) that builds one
     vLLM engine (full precision, revision-pinned, tensor-parallel per scale) and
     generates all (question x condition) prompts in one continuous-batched call
     at temperature=0, 1 trial (pre-reg section 6). Per-checkpoint subprocess
     isolation is REQUIRED: vLLM only releases GPU memory on process exit, so
     looping engines in a single process OOMs on the second checkpoint.

This entrypoint has NO Anthropic spend: extraction/verification/judging run
afterwards over the generated JSONL via the existing Phase B pipeline.

Usage (on the rented GPU box, after `pip install vllm`). The two env vars are
required on this stack: VLLM_USE_FLASHINFER_SAMPLER=0 (the box has no nvcc to
JIT-build flashinfer's sampler; neutral at temp=0) and ninja on PATH:
    PATH=$HOME/.local/bin:$PATH VLLM_USE_FLASHINFER_SAMPLER=0 HF_TOKEN=<tok> \
      python -m src.run_phase_c \
        --questions data/questions/frozen_900q_legal_amended.jsonl \
        --output-dir data/generations/phase_c \
        --only-families tulu3_8b olmo2_13b    # stage 8B+13B before the 70B run
Each checkpoint subprocess inherits this environment.
"""
from __future__ import annotations

import argparse
import functools
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from src.elicit import (
    GenerationSpec,
    PromptCondition,
    Question,
    VLLMGenerator,
    generate_offline_batch,
)
from src.pilot import CONDITIONS
from src.run_phase_b import load_model_registry, load_phase_b_questions

CHECKPOINTS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "models" / "phase_c_checkpoints.jsonl"
)

# Open-weights trial design (pre-reg section 6): deterministic decoding, 1 trial.
_TEMPERATURE = 0.0
_MAX_OUTPUT_TOKENS = 1024
# Full precision: 4-bit/8-bit quantization can shift hallucination rates, so the
# primary E1/E2 numbers are never quantized (plan section 5).
_DTYPE = "bfloat16"
_DEFAULT_SEED = 20260614
# Bounded context: our prompts are short (<= ~1500 tok incl. the 1024-token
# output), far under this. Setting it explicitly relieves KV-cache pressure
# (esp. 70B on 2x80GB) and is neutral at temperature=0 (greedy output unchanged,
# so it matches data generated under vLLM's larger default).
_MAX_MODEL_LEN = 4096
# Extra KV headroom for large multi-GPU checkpoints: 70B weights are ~70GB/GPU on
# 2x80GB, leaving little for KV cache at vLLM's default utilization.
_LARGE_GPU_MEMORY_UTILIZATION = 0.96

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _spec_from_checkpoint_row(row: dict[str, Any], seed: int) -> GenerationSpec:
    return GenerationSpec(
        model_id=row["model_id"],
        provider="vllm_local",
        temperature=_TEMPERATURE,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
        extra={
            "seed": seed,
            "apply_chat_template": row["apply_chat_template"],
            "revision": row["revision"],
        },
    )


def _engine_kwargs_from_row(row: dict[str, Any]) -> dict[str, Any]:
    """vLLM `LLM(...)` constructor kwargs: pinned revision, TP degree, dtype,
    bounded context, and KV headroom for multi-GPU large checkpoints."""
    kwargs: dict[str, Any] = {
        "revision": row["revision"],
        "tensor_parallel_size": row["tensor_parallel_size"],
        "dtype": _DTYPE,
        "max_model_len": _MAX_MODEL_LEN,
    }
    if row["tensor_parallel_size"] > 1:
        kwargs["gpu_memory_utilization"] = _LARGE_GPU_MEMORY_UTILIZATION
    return kwargs


def validate_checkpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fail fast on registry drift before any GPU is rented.

    Enforces the base-vs-post-trained invariant E1 depends on: a base checkpoint
    (stage_ordinal 0) is never post-trained and never gets a chat template.
    """
    seen: set[str] = set()
    for row in rows:
        model_id = row["model_id"]
        if model_id in seen:
            raise ValueError(f"duplicate model_id in checkpoint registry: {model_id}")
        seen.add(model_id)

        if not _SHA_RE.match(str(row["revision"])):
            raise ValueError(
                f"{model_id}: revision must be a 40-char hex commit SHA, "
                f"got {row['revision']!r}"
            )

        is_base = row["stage_ordinal"] == 0
        if is_base:
            if row["is_post_trained"] != 0:
                raise ValueError(
                    f"base checkpoint {model_id} must have is_post_trained=0, "
                    f"got {row['is_post_trained']}"
                )
            if row["apply_chat_template"]:
                raise ValueError(
                    f"base checkpoint {model_id} must not apply_chat_template "
                    f"(no chat template exists for a base model)"
                )
        else:
            if row["is_post_trained"] != 1:
                raise ValueError(
                    f"post-trained checkpoint {model_id} must have is_post_trained=1"
                )
            if not row["apply_chat_template"]:
                raise ValueError(
                    f"post-trained checkpoint {model_id} must apply_chat_template"
                )
    return rows


def _safe_filename(model_id: str) -> str:
    return model_id.replace("/", "__")


def _default_generator_factory(row: dict[str, Any]) -> VLLMGenerator:
    return VLLMGenerator(
        model_id=row["hf_repo"],
        apply_chat_template=row["apply_chat_template"],
        engine_kwargs=_engine_kwargs_from_row(row),
    )


def filter_by_family(
    rows: list[dict[str, Any]], only: list[str] | None
) -> list[dict[str, Any]]:
    if only is None:
        return rows
    requested = set(only)
    known = {row["model_family"] for row in rows}
    unknown = requested - known
    if unknown:
        raise ValueError(
            f"--only-families values not in registry: {sorted(unknown)}. "
            f"Registry has: {sorted(known)}"
        )
    return [row for row in rows if row["model_family"] in requested]


def _count_records(path: Path) -> int:
    """Number of JSONL records already written (0 if absent). Backs the
    idempotent skip so a complete checkpoint never reloads its multi-GB model."""
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run_one(
    row: dict[str, Any],
    questions: list[Question],
    conditions: list[PromptCondition],
    output_dir: Path,
    seed: int = _DEFAULT_SEED,
    generator_factory: Callable[[dict[str, Any]], Any] = _default_generator_factory,
) -> None:
    """Generate ONE checkpoint in the current process.

    run_sweep invokes this in a dedicated subprocess per checkpoint so vLLM fully
    releases GPU memory between models (its EngineCore subprocess + CUDA context
    only free on process exit; looping engines in one process OOMs on the 2nd).
    Skips if the output already holds every (question x condition) record.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{_safe_filename(row['model_id'])}.jsonl"
    if _count_records(out_path) >= len(questions) * len(conditions):
        return
    spec = _spec_from_checkpoint_row(row, seed)
    generator = generator_factory(row)
    generate_offline_batch(
        questions=questions,
        conditions=conditions,
        spec=spec,
        generator=generator,
        output_path=out_path,
    )


def _subprocess_spawn(
    row: dict[str, Any],
    *,
    questions_path: Path,
    checkpoints_path: Path,
    output_dir: Path,
    seed: int,
) -> None:
    """Default run_sweep spawn: run one checkpoint in a fresh process. Inherits
    the parent environment, so VLLM_USE_FLASHINFER_SAMPLER / HF_TOKEN / PATH set
    on the sweep command flow through to each checkpoint."""
    subprocess.run(
        [
            sys.executable, "-m", "src.run_phase_c",
            "--single-model", row["model_id"],
            "--questions", str(questions_path),
            "--checkpoints", str(checkpoints_path),
            "--output-dir", str(output_dir),
            "--seed", str(seed),
        ],
        check=True,
    )


def run_sweep(
    rows: list[dict[str, Any]],
    output_dir: Path,
    expected_per_model: int,
    spawn: Callable[[dict[str, Any]], None],
) -> None:
    """Run each checkpoint in its OWN process via `spawn(row)` so vLLM releases
    GPU memory between models. Skips checkpoints already complete on disk
    (idempotent resume), so a killed sweep restarts where it stopped."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        out_path = output_dir / f"{_safe_filename(row['model_id'])}.jsonl"
        if _count_records(out_path) >= expected_per_model:
            continue
        spawn(row)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="src.run_phase_c",
        description="Phase C open-weights E1+E2 confirmatory sweep (vLLM).",
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, default=CHECKPOINTS_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--only-families",
        nargs="+",
        default=None,
        help="Subset registry by model_family (e.g. tulu3_8b olmo2_13b).",
    )
    parser.add_argument(
        "--single-model",
        default=None,
        help="Run exactly one checkpoint (by model_id) in THIS process. Used "
             "internally by the sweep to isolate each checkpoint's vLLM engine.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    rows = validate_checkpoints(load_model_registry(args.checkpoints))
    questions = load_phase_b_questions(args.questions)
    conditions = list(CONDITIONS)

    if args.single_model is not None:
        row = next((r for r in rows if r["model_id"] == args.single_model), None)
        if row is None:
            raise SystemExit(f"--single-model {args.single_model!r} not in registry")
        run_one(row, questions, conditions, args.output_dir, seed=args.seed)
        return

    rows = filter_by_family(rows, args.only_families)
    spawn = functools.partial(
        _subprocess_spawn,
        questions_path=args.questions,
        checkpoints_path=args.checkpoints,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    run_sweep(rows, args.output_dir, len(questions) * len(conditions), spawn)


if __name__ == "__main__":
    main()
