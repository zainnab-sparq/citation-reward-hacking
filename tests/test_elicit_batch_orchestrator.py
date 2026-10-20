"""Tests for the Batch API orchestration layer in src/elicit.py.

Covers the manifest sidecar (ledger of submitted batches), submit/poll/fetch
helpers, and the top-level generate_via_batch_api() with resume-on-restart.

The pure-function layer (BatchKey, build_batch_requests, parse_batch_results)
is exercised in test_elicit_batch.py.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.elicit import (
    BatchKey,
    BatchManifestEntry,
    GenerationSpec,
    PromptCondition,
    Question,
    fetch_and_write_results,
    generate_via_batch_api,
    load_manifest,
    poll_until_ended,
    submit_batch,
    write_manifest_entry,
)


# ---------- Fixtures ----------


@pytest.fixture
def opus_spec() -> GenerationSpec:
    return GenerationSpec(
        model_id="claude-opus-4-7",
        provider="anthropic",
        temperature=1.0,
        max_output_tokens=512,
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
            id="academic_known_real_0002",
            domain="academic",
            text="What is dropout?",
            expected_existence="answerable",
        ),
    ]


@pytest.fixture
def neutral_only() -> list[PromptCondition]:
    return [PromptCondition(name="neutral", template="Q: {question}")]


def _succeeded_response(custom_id: str, text: str = "ok") -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[SimpleNamespace(type="text", text=text)],
                usage=SimpleNamespace(input_tokens=10, output_tokens=20),
                stop_reason="end_turn",
            ),
        ),
    )


class _FakeBatchesClient:
    """In-memory stand-in for AsyncAnthropic().messages.batches.

    Records calls; returns scripted responses for create/retrieve/results.
    """

    def __init__(
        self,
        *,
        create_batch_id: str = "msgbatch_test_01",
        retrieve_statuses: list[str] | None = None,
        results_payload: list[SimpleNamespace] | None = None,
    ) -> None:
        self.create_batch_id = create_batch_id
        # Default: one in_progress then ended (so the first poll loops once).
        self.retrieve_statuses = list(retrieve_statuses or ["in_progress", "ended"])
        self.results_payload = list(results_payload or [])
        self.create_calls: list[dict] = []
        self.retrieve_calls: list[str] = []
        self.results_calls: list[str] = []

    async def create(self, *, requests: list[dict]) -> SimpleNamespace:
        self.create_calls.append({"requests": list(requests)})
        return SimpleNamespace(
            id=self.create_batch_id,
            processing_status="in_progress",
        )

    async def retrieve(self, batch_id: str) -> SimpleNamespace:
        self.retrieve_calls.append(batch_id)
        if self.retrieve_statuses:
            status = self.retrieve_statuses.pop(0)
        else:
            status = "ended"
        return SimpleNamespace(id=batch_id, processing_status=status)

    async def results(self, batch_id: str) -> object:
        """Return an async-iterable of results (mimics AsyncJSONLDecoder)."""
        self.results_calls.append(batch_id)
        payload = list(self.results_payload)

        class _AsyncIter:
            def __init__(self, items: list) -> None:
                self._items = list(items)

            def __aiter__(self):
                return self

            async def __anext__(self):
                if not self._items:
                    raise StopAsyncIteration
                return self._items.pop(0)

        return _AsyncIter(payload)


class _FakeClient:
    def __init__(self, batches: _FakeBatchesClient) -> None:
        self.messages = SimpleNamespace(batches=batches)


# ---------- BatchManifestEntry ----------


def test_manifest_entry_round_trips_through_jsonl_dict() -> None:
    """to_jsonl_dict / from_jsonl_dict are inverses for all fields."""
    entry = BatchManifestEntry(
        batch_id="msgbatch_abc",
        model_id="claude-opus-4-7",
        output_path="data/generations/opus.jsonl",
        request_count=42,
        submitted_at="2026-05-18T10:00:00Z",
        ended_at=None,
        final_status=None,
    )

    serialized = entry.to_jsonl_dict()
    rehydrated = BatchManifestEntry.from_jsonl_dict(serialized)

    assert rehydrated == entry


def test_manifest_entry_serializes_to_pure_json_types() -> None:
    """to_jsonl_dict must json.dumps cleanly (no Path/datetime objects)."""
    entry = BatchManifestEntry(
        batch_id="msgbatch_xyz",
        model_id="claude-sonnet-4-6",
        output_path="data/generations/sonnet.jsonl",
        request_count=7,
        submitted_at="2026-05-18T10:00:00Z",
        ended_at="2026-05-18T11:00:00Z",
        final_status="ended",
    )

    raw = json.dumps(entry.to_jsonl_dict())
    decoded = json.loads(raw)
    assert decoded["batch_id"] == "msgbatch_xyz"
    assert decoded["final_status"] == "ended"


# ---------- write_manifest_entry / load_manifest ----------


def test_write_manifest_entry_appends_jsonl_line(tmp_path: Path) -> None:
    """write_manifest_entry creates the file on first call and appends on subsequent."""
    manifest_path = tmp_path / "batches.manifest.jsonl"
    entry1 = BatchManifestEntry(
        batch_id="msgbatch_001",
        model_id="claude-opus-4-7",
        output_path="out.jsonl",
        request_count=10,
        submitted_at="2026-05-18T10:00:00Z",
    )
    entry2 = BatchManifestEntry(
        batch_id="msgbatch_002",
        model_id="claude-opus-4-7",
        output_path="out.jsonl",
        request_count=5,
        submitted_at="2026-05-18T10:05:00Z",
    )

    write_manifest_entry(manifest_path, entry1)
    write_manifest_entry(manifest_path, entry2)

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["batch_id"] == "msgbatch_001"
    assert json.loads(lines[1])["batch_id"] == "msgbatch_002"


def test_load_manifest_reduces_by_batch_id_keeping_latest(tmp_path: Path) -> None:
    """Multiple rows with the same batch_id collapse to the most recent (ledger semantics)."""
    manifest_path = tmp_path / "batches.manifest.jsonl"
    submitted = BatchManifestEntry(
        batch_id="msgbatch_001",
        model_id="claude-opus-4-7",
        output_path="out.jsonl",
        request_count=10,
        submitted_at="2026-05-18T10:00:00Z",
    )
    ended = BatchManifestEntry(
        batch_id="msgbatch_001",
        model_id="claude-opus-4-7",
        output_path="out.jsonl",
        request_count=10,
        submitted_at="2026-05-18T10:00:00Z",
        ended_at="2026-05-18T11:00:00Z",
        final_status="ended",
    )
    write_manifest_entry(manifest_path, submitted)
    write_manifest_entry(manifest_path, ended)

    entries = load_manifest(manifest_path)

    assert len(entries) == 1
    assert entries[0].final_status == "ended"
    assert entries[0].ended_at == "2026-05-18T11:00:00Z"


def test_load_manifest_returns_empty_for_missing_file(tmp_path: Path) -> None:
    """A non-existent manifest is treated as 'no batches yet' (cold start)."""
    assert load_manifest(tmp_path / "does-not-exist.jsonl") == []


# ---------- submit_batch ----------


def test_submit_batch_calls_create_with_requests_and_writes_manifest(
    tmp_path: Path,
    two_questions: list[Question],
    neutral_only: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """submit_batch posts to client.messages.batches.create AND records the
    batch in the manifest sidecar."""
    batches = _FakeBatchesClient(create_batch_id="msgbatch_submitted_01")
    client = _FakeClient(batches)
    manifest_path = tmp_path / "batches.manifest.jsonl"
    output_path = tmp_path / "out.jsonl"

    entry = asyncio.run(
        submit_batch(
            client=client,
            questions=two_questions,
            conditions=neutral_only,
            spec=opus_spec,
            num_trials=1,
            manifest_path=manifest_path,
            output_path=output_path,
        )
    )

    assert len(batches.create_calls) == 1
    sent = batches.create_calls[0]["requests"]
    assert len(sent) == 2  # 2 questions × 1 condition × 1 trial
    assert all(r["params"]["model"] == "claude-opus-4-7" for r in sent)

    assert entry.batch_id == "msgbatch_submitted_01"
    assert entry.request_count == 2
    assert entry.model_id == "claude-opus-4-7"

    entries = load_manifest(manifest_path)
    assert len(entries) == 1
    assert entries[0].batch_id == "msgbatch_submitted_01"


# ---------- poll_until_ended ----------


def test_poll_until_ended_returns_when_status_ends() -> None:
    """Polling stops as soon as processing_status == 'ended'."""
    batches = _FakeBatchesClient(
        retrieve_statuses=["in_progress", "in_progress", "ended"]
    )
    client = _FakeClient(batches)

    final = asyncio.run(
        poll_until_ended(client=client, batch_id="msgbatch_xyz", poll_interval_s=0)
    )

    assert final.processing_status == "ended"
    assert batches.retrieve_calls == ["msgbatch_xyz", "msgbatch_xyz", "msgbatch_xyz"]


def test_poll_until_ended_short_circuits_when_already_ended() -> None:
    """If the batch is already 'ended' on first poll, return immediately."""
    batches = _FakeBatchesClient(retrieve_statuses=["ended"])
    client = _FakeClient(batches)

    final = asyncio.run(
        poll_until_ended(client=client, batch_id="msgbatch_xyz", poll_interval_s=0)
    )

    assert final.processing_status == "ended"
    assert len(batches.retrieve_calls) == 1


def test_poll_until_ended_raises_on_timeout() -> None:
    """If poll exceeds the timeout, raise TimeoutError (caller can decide)."""
    # All polls return in_progress; tight timeout forces failure.
    batches = _FakeBatchesClient(retrieve_statuses=["in_progress"] * 100)
    client = _FakeClient(batches)

    with pytest.raises(TimeoutError):
        asyncio.run(
            poll_until_ended(
                client=client,
                batch_id="msgbatch_stuck",
                poll_interval_s=0.01,
                timeout_s=0.05,
            )
        )


# ---------- fetch_and_write_results ----------


def test_fetch_and_write_results_appends_jsonl(
    tmp_path: Path, opus_spec: GenerationSpec
) -> None:
    """fetch_and_write_results streams results into the output JSONL."""
    cid_a = BatchKey("academic_known_real_0001", opus_spec.model_id, "neutral", 0).encode()
    cid_b = BatchKey("academic_known_real_0002", opus_spec.model_id, "neutral", 0).encode()
    batches = _FakeBatchesClient(
        results_payload=[
            _succeeded_response(cid_a, "answer-A"),
            _succeeded_response(cid_b, "answer-B"),
        ]
    )
    client = _FakeClient(batches)
    output_path = tmp_path / "out.jsonl"

    asyncio.run(
        fetch_and_write_results(
            client=client, batch_id="msgbatch_done", output_path=output_path
        )
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert {r["output_text"] for r in records} == {"answer-A", "answer-B"}
    assert all(r["model_id"] == opus_spec.model_id for r in records)


def test_fetch_and_write_results_skips_already_written_keys(
    tmp_path: Path, opus_spec: GenerationSpec
) -> None:
    """Idempotency: re-running fetch after partial write doesn't duplicate rows."""
    cid_a = BatchKey("academic_known_real_0001", opus_spec.model_id, "neutral", 0).encode()
    cid_b = BatchKey("academic_known_real_0002", opus_spec.model_id, "neutral", 0).encode()
    output_path = tmp_path / "out.jsonl"
    # Pre-populate output with q1 already written.
    output_path.write_text(
        json.dumps(
            {
                "question_id": "academic_known_real_0001",
                "condition": "neutral",
                "model_id": opus_spec.model_id,
                "trial_id": 0,
                "output_text": "previous-A",
                "input_tokens": 1,
                "output_tokens": 1,
                "finish_reason": "end_turn",
                "timestamp": "2026-05-17T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    batches = _FakeBatchesClient(
        results_payload=[
            _succeeded_response(cid_a, "duplicate-A"),
            _succeeded_response(cid_b, "answer-B"),
        ]
    )
    client = _FakeClient(batches)

    asyncio.run(
        fetch_and_write_results(
            client=client, batch_id="msgbatch_done", output_path=output_path
        )
    )

    lines = output_path.read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    # 1 pre-existing q1 row + 1 new q2 row.
    assert len(records) == 2
    q1_rows = [r for r in records if r["question_id"] == "academic_known_real_0001"]
    assert len(q1_rows) == 1
    assert q1_rows[0]["output_text"] == "previous-A"  # NOT overwritten


# ---------- generate_via_batch_api (top-level) ----------


def test_generate_via_batch_api_end_to_end_cold_start(
    tmp_path: Path,
    two_questions: list[Question],
    neutral_only: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """Cold start: no manifest, no output. Submits, polls, fetches, writes."""
    cid_a = BatchKey("academic_known_real_0001", opus_spec.model_id, "neutral", 0).encode()
    cid_b = BatchKey("academic_known_real_0002", opus_spec.model_id, "neutral", 0).encode()
    batches = _FakeBatchesClient(
        create_batch_id="msgbatch_cold_01",
        retrieve_statuses=["ended"],
        results_payload=[
            _succeeded_response(cid_a, "A"),
            _succeeded_response(cid_b, "B"),
        ],
    )
    client = _FakeClient(batches)
    manifest_path = tmp_path / "batches.manifest.jsonl"
    output_path = tmp_path / "out.jsonl"

    asyncio.run(
        generate_via_batch_api(
            questions=two_questions,
            conditions=neutral_only,
            spec=opus_spec,
            client=client,
            output_path=output_path,
            manifest_path=manifest_path,
            num_trials=1,
            poll_interval_s=0,
        )
    )

    # Submitted once.
    assert len(batches.create_calls) == 1
    # Polled and fetched.
    assert len(batches.retrieve_calls) == 1
    assert len(batches.results_calls) == 1
    # Output written.
    assert output_path.exists()
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert {r["output_text"] for r in records} == {"A", "B"}
    # Manifest closed with 'ended' status.
    entries = load_manifest(manifest_path)
    assert len(entries) == 1
    assert entries[0].final_status == "ended"


def test_generate_via_batch_api_resumes_in_progress_batch(
    tmp_path: Path,
    two_questions: list[Question],
    neutral_only: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """Restart with manifest showing an in-progress batch: don't resubmit,
    just poll + fetch the existing batch_id."""
    manifest_path = tmp_path / "batches.manifest.jsonl"
    output_path = tmp_path / "out.jsonl"

    # Pre-existing manifest: batch was submitted but not yet ended.
    write_manifest_entry(
        manifest_path,
        BatchManifestEntry(
            batch_id="msgbatch_resumed_01",
            model_id=opus_spec.model_id,
            output_path=str(output_path),
            request_count=2,
            submitted_at="2026-05-18T09:00:00Z",
        ),
    )

    cid_a = BatchKey("academic_known_real_0001", opus_spec.model_id, "neutral", 0).encode()
    cid_b = BatchKey("academic_known_real_0002", opus_spec.model_id, "neutral", 0).encode()
    batches = _FakeBatchesClient(
        retrieve_statuses=["ended"],
        results_payload=[
            _succeeded_response(cid_a, "A"),
            _succeeded_response(cid_b, "B"),
        ],
    )
    client = _FakeClient(batches)

    asyncio.run(
        generate_via_batch_api(
            questions=two_questions,
            conditions=neutral_only,
            spec=opus_spec,
            client=client,
            output_path=output_path,
            manifest_path=manifest_path,
            num_trials=1,
            poll_interval_s=0,
        )
    )

    # NO new batch was submitted.
    assert len(batches.create_calls) == 0
    # The existing batch_id was polled and fetched.
    assert batches.retrieve_calls == ["msgbatch_resumed_01"]
    assert batches.results_calls == ["msgbatch_resumed_01"]
    # Manifest updated with ended status.
    entries = load_manifest(manifest_path)
    assert len(entries) == 1
    assert entries[0].batch_id == "msgbatch_resumed_01"
    assert entries[0].final_status == "ended"


def test_submit_batch_rejects_empty_requests(
    tmp_path: Path,
    opus_spec: GenerationSpec,
) -> None:
    """If no work is requested, submit_batch raises ValueError BEFORE the API call.

    Defends against silent failures from passing empty lists; without the guard
    the SDK rejects 0-request batches with an opaque server error.
    """
    batches = _FakeBatchesClient()
    client = _FakeClient(batches)
    manifest_path = tmp_path / "batches.manifest.jsonl"
    output_path = tmp_path / "out.jsonl"

    with pytest.raises(ValueError):
        asyncio.run(
            submit_batch(
                client=client,
                questions=[],
                conditions=[PromptCondition(name="neutral", template="Q: {question}")],
                spec=opus_spec,
                num_trials=1,
                manifest_path=manifest_path,
                output_path=output_path,
            )
        )

    assert len(batches.create_calls) == 0


def test_generate_via_batch_api_picks_latest_in_progress_batch(
    tmp_path: Path,
    two_questions: list[Question],
    neutral_only: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """When the manifest has multiple in-progress entries for the same
    (model_id, output_path), resume the LATEST (newest in file).

    Without this guarantee, a crash mid-run + restart could resume an
    abandoned older batch while the newer one keeps running on Anthropic's
    side.
    """
    manifest_path = tmp_path / "batches.manifest.jsonl"
    output_path = tmp_path / "out.jsonl"

    older = BatchManifestEntry(
        batch_id="msgbatch_older",
        model_id=opus_spec.model_id,
        output_path=str(output_path),
        request_count=2,
        submitted_at="2026-05-18T09:00:00Z",
    )
    newer = BatchManifestEntry(
        batch_id="msgbatch_newer",
        model_id=opus_spec.model_id,
        output_path=str(output_path),
        request_count=2,
        submitted_at="2026-05-18T09:30:00Z",
    )
    write_manifest_entry(manifest_path, older)
    write_manifest_entry(manifest_path, newer)

    cid_a = BatchKey("academic_known_real_0001", opus_spec.model_id, "neutral", 0).encode()
    cid_b = BatchKey("academic_known_real_0002", opus_spec.model_id, "neutral", 0).encode()
    batches = _FakeBatchesClient(
        retrieve_statuses=["ended"],
        results_payload=[
            _succeeded_response(cid_a, "A"),
            _succeeded_response(cid_b, "B"),
        ],
    )
    client = _FakeClient(batches)

    asyncio.run(
        generate_via_batch_api(
            questions=two_questions,
            conditions=neutral_only,
            spec=opus_spec,
            client=client,
            output_path=output_path,
            manifest_path=manifest_path,
            num_trials=1,
            poll_interval_s=0,
        )
    )

    # Resumed the newer batch, not the older.
    assert batches.retrieve_calls == ["msgbatch_newer"]
    assert batches.results_calls == ["msgbatch_newer"]
    assert len(batches.create_calls) == 0


def test_generate_via_batch_api_skips_if_all_outputs_already_present(
    tmp_path: Path,
    two_questions: list[Question],
    neutral_only: list[PromptCondition],
    opus_spec: GenerationSpec,
) -> None:
    """If every (q, c, trial) is already in output, don't submit anything."""
    output_path = tmp_path / "out.jsonl"
    manifest_path = tmp_path / "batches.manifest.jsonl"
    # Pre-populate output with both expected (q, c, trial) records.
    output_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": q.id,
                    "condition": "neutral",
                    "model_id": opus_spec.model_id,
                    "trial_id": 0,
                    "output_text": "already-present",
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "finish_reason": "end_turn",
                    "timestamp": "2026-05-17T00:00:00Z",
                }
            )
            for q in two_questions
        )
        + "\n",
        encoding="utf-8",
    )
    batches = _FakeBatchesClient()
    client = _FakeClient(batches)

    asyncio.run(
        generate_via_batch_api(
            questions=two_questions,
            conditions=neutral_only,
            spec=opus_spec,
            client=client,
            output_path=output_path,
            manifest_path=manifest_path,
            num_trials=1,
            poll_interval_s=0,
        )
    )

    assert len(batches.create_calls) == 0
    assert len(batches.retrieve_calls) == 0
    assert len(batches.results_calls) == 0
