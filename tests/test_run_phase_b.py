"""Contract tests for src/run_phase_b.py (Phase B CLI orchestrator).

The orchestrator's only job is to glue these three things together:
  (a) load the frozen 900-question JSONL into Question objects (handling
      the 'question' field name + preserving 'cell' / 'expected_citation'
      in Question.metadata for the aggregation join);
  (b) load the Phase B model registry (data/models/phase_b_models.jsonl);
  (c) for each model in the (optionally --models-filtered) registry, call
      generate_via_batch_api with the four pilot CONDITIONS and the per-model
      output file.

Generation, polling, and result-fetching all live in src/elicit.py and are
exercised in test_elicit_batch_orchestrator.py — this module only tests
that the orchestrator hands the right args to generate_via_batch_api.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.elicit import Question


# ---------- load_phase_b_questions ----------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


@pytest.fixture
def frozen_question_record() -> dict:
    return {
        "id": "academic_known_real_0000",
        "domain": "academic",
        "question": "What foundational paper established construct validity?",
        "expected_existence": "answerable",
        "cell": "known_real",
        "expected_citation": {"title": "Construct validity", "year": 1955},
        "metadata": {"recruited_via": "openalex", "seed": 42},
    }


def test_load_phase_b_questions_maps_question_field_to_text(
    tmp_path: Path, frozen_question_record: dict
):
    """Frozen 900q JSONL uses 'question' field; Question dataclass uses 'text'."""
    from src.run_phase_b import load_phase_b_questions

    path = tmp_path / "q.jsonl"
    _write_jsonl(path, [frozen_question_record])

    qs = load_phase_b_questions(path)
    assert len(qs) == 1
    q = qs[0]
    assert isinstance(q, Question)
    assert q.id == "academic_known_real_0000"
    assert q.domain == "academic"
    assert q.text == "What foundational paper established construct validity?"
    assert q.expected_existence == "answerable"


def test_load_phase_b_questions_preserves_cell_and_expected_citation_in_metadata(
    tmp_path: Path, frozen_question_record: dict
):
    """Cell + expected_citation must survive into Question.metadata so the
    aggregation join (src/analysis/aggregate.py) can recover them."""
    from src.run_phase_b import load_phase_b_questions

    path = tmp_path / "q.jsonl"
    _write_jsonl(path, [frozen_question_record])

    q = load_phase_b_questions(path)[0]
    assert q.metadata["cell"] == "known_real"
    assert q.metadata["expected_citation"]["year"] == 1955
    # Original metadata still present (recruiter audit trail).
    assert q.metadata["recruited_via"] == "openalex"
    assert q.metadata["seed"] == 42


def test_load_phase_b_questions_skips_blank_lines(
    tmp_path: Path, frozen_question_record: dict
):
    from src.run_phase_b import load_phase_b_questions

    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps(frozen_question_record) + "\n\n", encoding="utf-8"
    )
    qs = load_phase_b_questions(path)
    assert len(qs) == 1


# ---------- load_model_registry ----------


@pytest.fixture
def registry_records() -> list[dict]:
    return [
        {
            "model_id": "claude-haiku-4-5-20251001",
            "family": "anthropic",
            "is_post_trained": 1,
            "is_open_weights": 0,
            "phase_b_role": "step1_haiku_full_900q",
            "pin_kind": "dated_build",
        },
        {
            "model_id": "claude-opus-4-7",
            "family": "anthropic",
            "is_post_trained": 1,
            "is_open_weights": 0,
            "phase_b_role": "step3_opus_full_900q",
            "pin_kind": "alias",
        },
        {
            "model_id": "claude-sonnet-4-6",
            "family": "anthropic",
            "is_post_trained": 1,
            "is_open_weights": 0,
            "phase_b_role": "step3_sonnet_full_900q",
            "pin_kind": "alias",
        },
    ]


def test_load_model_registry_reads_all_rows(
    tmp_path: Path, registry_records: list[dict]
):
    from src.run_phase_b import load_model_registry

    path = tmp_path / "phase_b_models.jsonl"
    _write_jsonl(path, registry_records)

    registry = load_model_registry(path)
    assert len(registry) == 3
    assert [r["model_id"] for r in registry] == [
        "claude-haiku-4-5-20251001",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
    ]


# ---------- filter_registry ----------


def test_filter_registry_returns_all_when_filter_is_none(
    registry_records: list[dict],
):
    from src.run_phase_b import filter_registry

    assert filter_registry(registry_records, None) == registry_records


def test_filter_registry_subsets_by_model_id(registry_records: list[dict]):
    from src.run_phase_b import filter_registry

    out = filter_registry(registry_records, ["claude-haiku-4-5-20251001"])
    assert len(out) == 1
    assert out[0]["model_id"] == "claude-haiku-4-5-20251001"


def test_filter_registry_raises_on_unknown_model(registry_records: list[dict]):
    from src.run_phase_b import filter_registry

    with pytest.raises(ValueError, match="not in registry"):
        filter_registry(registry_records, ["claude-fictional-9-9"])


def test_filter_registry_preserves_input_order(registry_records: list[dict]):
    """Registry order matters for the Haiku-first staging in §16."""
    from src.run_phase_b import filter_registry

    out = filter_registry(
        registry_records,
        ["claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
    )
    # Output follows registry order (Haiku first), not request order.
    assert [r["model_id"] for r in out] == [
        "claude-haiku-4-5-20251001",
        "claude-sonnet-4-6",
    ]


# ---------- run() — hands correct args to generate_via_batch_api ----------


class _FakeGenerateBatch:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> None:
        self.calls.append(kwargs)


@pytest.mark.asyncio
async def test_run_calls_generate_via_batch_api_once_per_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_question_record: dict,
    registry_records: list[dict],
):
    from src import run_phase_b

    fake = _FakeGenerateBatch()
    monkeypatch.setattr(run_phase_b, "generate_via_batch_api", fake)

    questions_path = tmp_path / "q.jsonl"
    _write_jsonl(questions_path, [frozen_question_record])
    registry_path = tmp_path / "registry.jsonl"
    _write_jsonl(registry_path, registry_records)
    output_dir = tmp_path / "gens"
    manifest = tmp_path / "manifest.jsonl"

    await run_phase_b.run(
        questions_path=questions_path,
        registry_path=registry_path,
        output_dir=output_dir,
        manifest_path=manifest,
        trials=5,
        only_models=None,
        client=object(),  # fake_generate_batch ignores it
    )

    assert len(fake.calls) == 3
    model_ids = [c["spec"].model_id for c in fake.calls]
    assert model_ids == [
        "claude-haiku-4-5-20251001",
        "claude-opus-4-7",
        "claude-sonnet-4-6",
    ]


@pytest.mark.asyncio
async def test_run_passes_four_conditions_and_correct_trial_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_question_record: dict,
    registry_records: list[dict],
):
    from src import run_phase_b

    fake = _FakeGenerateBatch()
    monkeypatch.setattr(run_phase_b, "generate_via_batch_api", fake)

    questions_path = tmp_path / "q.jsonl"
    _write_jsonl(questions_path, [frozen_question_record])
    registry_path = tmp_path / "registry.jsonl"
    _write_jsonl(registry_path, registry_records[:1])  # just Haiku
    output_dir = tmp_path / "gens"
    manifest = tmp_path / "manifest.jsonl"

    await run_phase_b.run(
        questions_path=questions_path,
        registry_path=registry_path,
        output_dir=output_dir,
        manifest_path=manifest,
        trials=5,
        only_models=None,
        client=object(),
    )

    call = fake.calls[0]
    assert call["num_trials"] == 5
    condition_names = {c.name for c in call["conditions"]}
    assert condition_names == {
        "neutral",
        "citations_expected",
        "abstention_permitted",
        "length_matched",
    }


@pytest.mark.asyncio
async def test_run_writes_per_model_output_path_under_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_question_record: dict,
    registry_records: list[dict],
):
    from src import run_phase_b

    fake = _FakeGenerateBatch()
    monkeypatch.setattr(run_phase_b, "generate_via_batch_api", fake)

    questions_path = tmp_path / "q.jsonl"
    _write_jsonl(questions_path, [frozen_question_record])
    registry_path = tmp_path / "registry.jsonl"
    _write_jsonl(registry_path, registry_records[:1])
    output_dir = tmp_path / "gens"
    manifest = tmp_path / "manifest.jsonl"

    await run_phase_b.run(
        questions_path=questions_path,
        registry_path=registry_path,
        output_dir=output_dir,
        manifest_path=manifest,
        trials=1,
        only_models=None,
        client=object(),
    )

    call = fake.calls[0]
    assert call["output_path"] == output_dir / "claude-haiku-4-5-20251001.jsonl"
    assert call["manifest_path"] == manifest


@pytest.mark.asyncio
async def test_run_only_models_filter_subsets_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_question_record: dict,
    registry_records: list[dict],
):
    from src import run_phase_b

    fake = _FakeGenerateBatch()
    monkeypatch.setattr(run_phase_b, "generate_via_batch_api", fake)

    questions_path = tmp_path / "q.jsonl"
    _write_jsonl(questions_path, [frozen_question_record])
    registry_path = tmp_path / "registry.jsonl"
    _write_jsonl(registry_path, registry_records)
    output_dir = tmp_path / "gens"
    manifest = tmp_path / "manifest.jsonl"

    await run_phase_b.run(
        questions_path=questions_path,
        registry_path=registry_path,
        output_dir=output_dir,
        manifest_path=manifest,
        trials=5,
        only_models=["claude-haiku-4-5-20251001"],
        client=object(),
    )

    assert len(fake.calls) == 1
    assert fake.calls[0]["spec"].model_id == "claude-haiku-4-5-20251001"


# ---------- CLI argparse ----------


def test_parse_args_defaults_and_required(monkeypatch: pytest.MonkeyPatch):
    from src.run_phase_b import parse_args

    args = parse_args(
        [
            "--questions",
            "data/questions/frozen_900q_legal_amended.jsonl",
            "--registry",
            "data/models/phase_b_models.jsonl",
            "--output-dir",
            "data/generations/phase_b",
            "--manifest",
            "data/generations/phase_b/batches.manifest.jsonl",
        ]
    )
    assert args.questions == Path("data/questions/frozen_900q_legal_amended.jsonl")
    assert args.registry == Path("data/models/phase_b_models.jsonl")
    assert args.output_dir == Path("data/generations/phase_b")
    assert args.manifest == Path("data/generations/phase_b/batches.manifest.jsonl")
    assert args.trials == 5  # §16 default for Anthropic
    assert args.only_models is None


def test_parse_args_only_models_accepts_multi():
    from src.run_phase_b import parse_args

    args = parse_args(
        [
            "--questions",
            "q.jsonl",
            "--registry",
            "r.jsonl",
            "--output-dir",
            "o",
            "--manifest",
            "m.jsonl",
            "--only-models",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-7",
        ]
    )
    assert args.only_models == [
        "claude-haiku-4-5-20251001",
        "claude-opus-4-7",
    ]
