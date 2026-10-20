"""Contract tests for src/verify_batch.py (Phase B verification runner).

Generalizes pilot_reverify.reverify() into a Phase B runner that:
  - keys verification records by (question_id, condition, model_id, trial_id)
    (the pilot keyed by (qid, condition) because it ran one trial; Phase B
     runs 5 trials per (qid, condition, model), so the key must include both),
  - emits one VerificationRecord per generation (1:1 with extractions),
  - resumes on restart by skipping records already present in the output,
  - routes by domain inside verify_citation (legal -> CourtListener first).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.elicit import Question


@pytest.fixture
def two_questions() -> list[Question]:
    return [
        Question(
            id="academic_known_real_0000",
            domain="academic",
            text="What introduced the Transformer?",
            expected_existence="answerable",
            metadata={"cell": "known_real"},
        ),
        Question(
            id="legal_unanswerable_0001",
            domain="legal",
            text="Cite case ...",
            expected_existence="unanswerable",
            metadata={"cell": "unanswerable"},
        ),
    ]


@pytest.fixture
def two_extractions() -> list[dict]:
    return [
        {
            "question_id": "academic_known_real_0000",
            "condition": "citations_expected",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "extractor_model_id": "claude-sonnet-4-6",
            "extracted_at": "2026-06-04T05:00:00+00:00",
            "citations": [
                {"raw_span": "Vaswani 2017", "title": "Attention is all you need",
                 "authors": ["Vaswani"], "year": 2017, "venue": "NeurIPS"},
                {"raw_span": "Devlin 2019", "title": "BERT",
                 "authors": ["Devlin"], "year": 2019, "venue": "NAACL"},
            ],
        },
        {
            "question_id": "legal_unanswerable_0001",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 3,
            "extractor_model_id": "claude-sonnet-4-6",
            "extracted_at": "2026-06-04T05:00:00+00:00",
            "citations": [
                {"raw_span": "Ghost v. State, 99 F.4th 8847",
                 "title": "Ghost v. State", "case_name": "Ghost v. State",
                 "reporter": "99 F.4th 8847", "year": 2024, "authors": []},
            ],
        },
    ]


# ---------- VerificationRecord schema ----------


def test_verification_record_to_jsonl_dict_roundtrips():
    from src.verify_batch import VerificationRecord

    record = VerificationRecord(
        question_id="academic_known_real_0000",
        condition="citations_expected",
        model_id="claude-haiku-4-5-20251001",
        trial_id=2,
        n_citations=2,
        citation_statuses=["REAL", "FABRICATED"],
        verified_at="2026-06-04T05:30:00+00:00",
    )
    d = record.to_jsonl_dict()
    assert d["n_citations"] == 2
    assert d["citation_statuses"] == ["REAL", "FABRICATED"]
    assert d["trial_id"] == 2


# ---------- load_extractions ----------


def test_load_extractions_reads_jsonl(tmp_path: Path, two_extractions):
    from src.verify_batch import load_extractions

    path = tmp_path / "extr.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in two_extractions) + "\n",
        encoding="utf-8",
    )
    out = load_extractions(path)
    assert len(out) == 2
    assert out[0]["question_id"] == "academic_known_real_0000"


# ---------- verify_extractions: core logic ----------


class _FakeVerify:
    """Stand-in for src.verify.verify_citation.

    Returns canned statuses keyed by (raw_text, domain). Tests register
    expected calls; this object records every actual call for assertions.
    """

    def __init__(self, status_for: dict[tuple[str, str], str]):
        self._status_for = status_for
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, raw_text, *, domain, cache, parsed=None):
        from types import SimpleNamespace

        self.calls.append((raw_text, domain))
        status_str = self._status_for.get((raw_text, domain), "UNVERIFIABLE")
        # Mimic VerificationResult's .status.value access path used by
        # verify_extractions — full dataclass construction isn't needed.
        return SimpleNamespace(status=SimpleNamespace(value=status_str))


@pytest.mark.asyncio
async def test_verify_extractions_emits_one_record_per_extraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    two_questions,
    two_extractions,
):
    from src import verify_batch

    fake = _FakeVerify(
        {
            ("Vaswani 2017", "academic"): "REAL",
            ("Devlin 2019", "academic"): "REAL",
            ("Ghost v. State, 99 F.4th 8847", "legal"): "FABRICATED",
        }
    )
    monkeypatch.setattr(verify_batch, "verify_citation", fake)

    qmap = {q.id: q for q in two_questions}
    records = []
    async for r in verify_batch.verify_extractions(
        extractions=two_extractions,
        qmap=qmap,
        cache_path=tmp_path / "cache.jsonl",
        verified_at="2026-06-04T05:30:00+00:00",
    ):
        records.append(r)
    assert len(records) == 2


@pytest.mark.asyncio
async def test_verify_extractions_records_statuses_in_extracted_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    two_questions,
    two_extractions,
):
    from src import verify_batch

    fake = _FakeVerify(
        {
            ("Vaswani 2017", "academic"): "REAL",
            ("Devlin 2019", "academic"): "FABRICATED",
            ("Ghost v. State, 99 F.4th 8847", "legal"): "FABRICATED",
        }
    )
    monkeypatch.setattr(verify_batch, "verify_citation", fake)

    qmap = {q.id: q for q in two_questions}
    records = []
    async for r in verify_batch.verify_extractions(
        extractions=two_extractions,
        qmap=qmap,
        cache_path=tmp_path / "cache.jsonl",
        verified_at="2026-06-04T05:30:00+00:00",
    ):
        records.append(r)
    academic = next(r for r in records if r.question_id == "academic_known_real_0000")
    assert academic.citation_statuses == ["REAL", "FABRICATED"]
    legal = next(r for r in records if r.question_id == "legal_unanswerable_0001")
    assert legal.citation_statuses == ["FABRICATED"]
    assert legal.trial_id == 3


@pytest.mark.asyncio
async def test_verify_extractions_handles_zero_citation_extraction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, two_questions
):
    from src import verify_batch

    fake = _FakeVerify({})
    monkeypatch.setattr(verify_batch, "verify_citation", fake)

    extractions = [
        {
            "question_id": "academic_known_real_0000",
            "condition": "abstention_permitted",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "extractor_model_id": "claude-sonnet-4-6",
            "extracted_at": "2026-06-04T05:00:00+00:00",
            "citations": [],
        }
    ]
    qmap = {q.id: q for q in two_questions}
    records = []
    async for r in verify_batch.verify_extractions(
        extractions=extractions,
        qmap=qmap,
        cache_path=tmp_path / "cache.jsonl",
        verified_at="2026-06-04T05:30:00+00:00",
    ):
        records.append(r)
    assert len(records) == 1
    assert records[0].n_citations == 0
    assert records[0].citation_statuses == []
    assert fake.calls == []  # zero citations -> no verify calls


@pytest.mark.asyncio
async def test_verify_extractions_skips_error_extractions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, two_questions
):
    """Extraction records with an `error` field (failed Sonnet calls) carry no
    actionable citations; skip them so the output preserves valid-extraction
    semantics for the aggregator."""
    from src import verify_batch

    fake = _FakeVerify({})
    monkeypatch.setattr(verify_batch, "verify_citation", fake)

    extractions = [
        {
            "question_id": "academic_known_real_0000",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "extractor_model_id": "claude-sonnet-4-6",
            "extracted_at": "2026-06-04T05:00:00+00:00",
            "citations": [],
            "error": "rate",
        }
    ]
    qmap = {q.id: q for q in two_questions}
    records = [
        r
        async for r in verify_batch.verify_extractions(
            extractions=extractions,
            qmap=qmap,
            cache_path=tmp_path / "cache.jsonl",
            verified_at="2026-06-04T05:30:00+00:00",
        )
    ]
    assert records == []


@pytest.mark.asyncio
async def test_verify_extractions_raises_on_unknown_question(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, two_questions
):
    from src import verify_batch

    fake = _FakeVerify({})
    monkeypatch.setattr(verify_batch, "verify_citation", fake)

    extractions = [
        {
            "question_id": "ghost-q-999",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "trial_id": 0,
            "extractor_model_id": "claude-sonnet-4-6",
            "extracted_at": "2026-06-04T05:00:00+00:00",
            "citations": [],
        }
    ]
    qmap = {q.id: q for q in two_questions}
    with pytest.raises(KeyError, match="ghost-q-999"):
        async for _ in verify_batch.verify_extractions(
            extractions=extractions,
            qmap=qmap,
            cache_path=tmp_path / "cache.jsonl",
            verified_at="2026-06-04T05:30:00+00:00",
        ):
            pass


# ---------- write_verifications + resume-on-restart ----------


@pytest.mark.asyncio
async def test_run_writes_jsonl_and_resumes_on_restart(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    two_questions,
    two_extractions,
):
    from src import verify_batch

    fake = _FakeVerify(
        {
            ("Vaswani 2017", "academic"): "REAL",
            ("Devlin 2019", "academic"): "REAL",
            ("Ghost v. State, 99 F.4th 8847", "legal"): "FABRICATED",
        }
    )
    monkeypatch.setattr(verify_batch, "verify_citation", fake)

    extractions_path = tmp_path / "extr.jsonl"
    extractions_path.write_text(
        "\n".join(json.dumps(e) for e in two_extractions) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "verif.jsonl"

    qmap = {q.id: q for q in two_questions}
    await verify_batch.run(
        extractions_path=extractions_path,
        qmap=qmap,
        output_path=out_path,
        cache_path=tmp_path / "cache.jsonl",
    )

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first_call_count = len(fake.calls)
    assert first_call_count == 3  # 2 academic + 1 legal

    # Re-run; existing rows should NOT re-trigger verify_citation.
    await verify_batch.run(
        extractions_path=extractions_path,
        qmap=qmap,
        output_path=out_path,
        cache_path=tmp_path / "cache.jsonl",
    )
    assert len(fake.calls) == first_call_count
    # Still exactly 2 output rows; no duplicates.
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


# ---------- CLI argparse ----------


def test_parse_args_defaults_and_required():
    from src.verify_batch import parse_args

    args = parse_args(
        [
            "--extractions",
            "data/extractions/phase_b/claude-haiku-4-5-20251001.jsonl",
            "--questions",
            "data/questions/frozen_900q_legal_amended.jsonl",
            "--output",
            "data/verifications/phase_b/claude-haiku-4-5-20251001.jsonl",
        ]
    )
    assert args.extractions == Path(
        "data/extractions/phase_b/claude-haiku-4-5-20251001.jsonl"
    )
    assert args.questions == Path("data/questions/frozen_900q_legal_amended.jsonl")
    assert args.output == Path(
        "data/verifications/phase_b/claude-haiku-4-5-20251001.jsonl"
    )
    # Cache defaults to a Phase B cache file
    assert args.cache == Path("data/verifications/phase_b_cache.jsonl")
