# Briefing: Batch API Integration for elicit.py

## Section 1: Current elicit.py Architecture Map

**Classes & Functions:**
- `Question` (line 23–28, frozen dataclass): Holds question id, domain, text, answerability flag, metadata.
- `PromptCondition` (line 32–37, frozen dataclass): Wraps condition name + format template; `.render(question)` interpolates.
- `GenerationSpec` (line 41–47, frozen dataclass): Config for one generation run—model_id, provider, temperature, max_output_tokens, api_base.
- `GenerationRecord` (line 51–64, dataclass): Output record with question_id, condition, model_id, output_text, token counts, finish_reason, timestamp, optional error.
- `Generator` Protocol (line 67–70): Async interface all providers implement; `generate(prompt: str, spec: GenerationSpec) → dict[...]`.
- `AnthropicGenerator` (line 75–101): Concrete sync adapter for Anthropic Messages API; calls `client.messages.create()` (line 89), extracts text blocks and usage (lines 95–100).
- `_record_key()` (line 106–107): Tuple helper `(question_id, condition, model_id)` for deduplication.
- `_load_existing()` (line 110–119): Resume mechanism—reads output_path and rebuilds set of already-completed (question, condition, model) tuples; skips if file absent.
- `_now_iso()` (line 122–123): UTC timestamp formatter.
- `generate_batch()` (line 126–180): Main orchestrator. Creates output dir (line 140), loads existing keys (line 141), spawns async tasks (line 179) for each (question, condition) pair capped by concurrency semaphore (line 142). Checks if key in existing (line 147); if present, returns early (line 148). Otherwise calls generator.generate() (line 152), catches exceptions (line 163), writes one JSONL record per task (lines 176–177) under write_lock to prevent interleave.

**Resume-on-restart:** The sync path relies on file-based deduplication: before starting a task, `_load_existing()` parses all existing JSONL records and extracts their keys. If a key matches the current (question_id, condition, model_id), the task is skipped entirely—no duplicate calls are made, and no duplicate records are written.

**Test patterns (test_elicit.py):**
- `FakeGenerator` (line 17–30): Mock generator returning hardcoded dict with output_text, token counts, finish_reason.
- Fixtures: `question_neural_attention` (line 34–40), `conditions` (line 44–58), `fake_spec` (line 62–68).
- `test_generate_batch_produces_one_record_per_question_condition` (line 72–92): Runs one question × three conditions, validates three JSONL lines with unique condition names.
- `test_generate_batch_resumes_skipping_existing_records` (line 96–130): Pre-populates output file, then reruns; verifies that only two new records are written and generator is called twice (not three times).
- `test_anthropic_generator_calls_messages_api` (line 134–170): Monkeypatches AsyncAnthropic to capture payload; validates model, max_tokens, temperature, messages shape, and response parsing.

---

## Section 2: Anthropic Message Batches API Surface

**SDK Methods available on `AsyncAnthropic.messages.batches`:**
- `.create(requests=[...])` → returns batch object with id, processing_status, request_counts, created_at.
- `.retrieve(batch_id)` → polls status; returns same object with updated request_counts and processing_status.
- `.results(batch_id, limit=None)` → async iterator/paginated endpoint returning individual result objects keyed by custom_id.
- `.cancel(batch_id)` → cancels a batch still processing; completed requests are unaffected.
- `.list()` → returns paginated list of all batches for the account.

**Request Shape:**
```json
{
  "requests": [
    {
      "custom_id": "string (required, unique within batch)",
      "params": {
        "model": "claude-opus-4-7",
        "messages": [{"role": "user", "content": "..."}],
        "max_tokens": 1024,
        "temperature": 0.7,
        "system": "...",
        ...other Messages API fields
      }
    }
  ]
}
```
Snapshot pinning is specified in the model field (e.g., `"claude-opus-4-7-2025-05-15"`).

**Response Shape:**
- **Batch submission response**: `{id, type, processing_status, request_counts: {processing, succeeded, errored}, created_at}`.
- **Status polling**: Same structure; update processing_status field to track (processing → succeeded or failed).
- **Results**: JSONL stream via paginated endpoint; each line contains `{custom_id, result: {type, message: {...}, ...}, error: null}` on success or `{custom_id, error: {type, message}, result: null}` on per-request error.

**Polling pattern:** Call `.retrieve(batch_id)` in a loop until processing_status ≠ "processing"; typically poll every 10–30s. No webhook support; SDK provides convenience polling if needed.

**Gotchas:**
- Max batch size: 100,000 requests, 256MB total payload.
- Max processing time: 24h; requests not completed by then are marked errored.
- Results are unordered and must be matched via custom_id.
- Per-request errors do not fail the batch; the batch completes as succeeded even if some requests have errors. Check the error field in each result.
- Snapshot IDs must be included in the model field inside each request's params; batches do not accept a global model override.

---

## Section 3: Integration Design Sketch

**Recommended architecture:**
1. **New `AnthropicBatchProvider` class** (parallel to existing `AnthropicGenerator`) implementing the `Generator` Protocol. This adapter internally calls `client.messages.batches.create()` and polls/retrieves, but **blocks on batch completion before returning** (synchronous from the caller's perspective). This way, `generate_batch()` remains unchanged; the batch polling logic is encapsulated.

   Alternative (not recommended): Add a batch mode flag to `generate_batch()` that changes orchestration strategy entirely, but this duplicates code and complicates the codebase.

2. **`custom_id` encoding/decoding helper** (e.g., `_encode_batch_key(question_id, model_snapshot, condition, trial_id) → str` and reverse). This is used both at submission time (line 126–180, when building requests) and at result retrieval time (when mapping results back to questions). A single shared function ensures encoding/decoding stay paired.

3. **Batch persistence on disk:**
   - Write submitted batch IDs to a `.batch_ids.jsonl` file or similar, one per line: `{batch_id, timestamp_submitted, status}`.
   - On restart, load this file and call `.retrieve(batch_id)` for each; skip submission if a batch for the same (question set, condition set, model, trial) already exists in processing/succeeded state.
   - This prevents resubmission of already-queued work.

4. **Per-request errors:** Log and persist them in the output JSONL under the error field (as the sync path does at line 173). Do not retry per-request errors within the batch; the batch completes successfully even if individual requests fail. Caller can re-run only the failed (question, condition, trial) tuple on a subsequent sync pass if needed.

5. **Batch mode invocation:**
   - Add an optional `batch_mode: bool = False` parameter to `generate_batch()`.
   - If True, instantiate `AnthropicBatchProvider()` instead of `AnthropicGenerator()`.
   - If False (default), use the existing sync path for backwards compatibility.

**Open design question:** Should the batch polling happen inside `AnthropicBatchProvider.generate()` (blocking, single-request synchronous interface) or should `generate_batch()` itself detect batch mode and implement a separate polling loop? Recommendation: block inside the adapter. This keeps orchestration logic unified and allows sync and batch to coexist cleanly.

---

## Section 4: Test Strategy Outline

**Mocking strategy:**
The existing tests use `monkeypatch` to replace client internals (see test_anthropic_generator_calls_messages_api, line 134). For batches, mock the async client's `messages.batches` methods. `respx` can mock HTTP endpoints if needed, but for unit tests, direct method mocking is simpler. The SDK's batch results are returned as async iterables, so mock with `AsyncIterator` or `AsyncGenerator`.

**Failing tests to write first (TDD cycle):**

1. **`test_batch_provider_submit_returns_batch_id_and_timestamp`**
   - Create an AnthropicBatchProvider with a mocked client.
   - Call generate() with a prompt and spec.
   - Mock client.messages.batches.create() to return `{id: "batch_123", ...}` and mock .retrieve() to return processing_status="succeeded".
   - Assert batch ID is returned (or persisted), submission timestamp is logged.

2. **`test_batch_provider_custom_id_encoding_preserves_question_condition_model_trial`**
   - Call `_encode_batch_key("q-001", "claude-opus-4-7-2025-05-15", "citations_expected", 0)`.
   - Decode result and verify all four components round-trip.

3. **`test_batch_provider_resume_skips_already_submitted_batch`**
   - Pre-populate batch ID cache with a batch for (q-001, condition, model, trial).
   - Mock client.messages.batches.list() to return that batch in succeeded state.
   - Call generate_batch() with same questions/conditions/spec.
   - Verify client.messages.batches.create() is NOT called; instead .retrieve() is called on the cached ID.

4. **`test_batch_provider_polls_until_succeeded`**
   - Mock .retrieve() to return processing_status="processing" twice, then processing_status="succeeded".
   - Assert polling loop runs at least 3 times before exiting.

5. **`test_batch_provider_per_request_error_is_logged_not_retried`**
   - Mock .results() to return one record with `{custom_id: "...", error: {message: "..."}, result: null}`.
   - Assert the output record has error field populated; verify generator.generate() is not called a second time for that request.

6. **`test_batch_provider_snapshot_id_in_model_field`**
   - Mock client.messages.batches.create() and capture the requests list.
   - Verify each request's params.model == the expected dated snapshot (e.g., "claude-opus-4-7-2025-05-15"), not just the base model name.

---

## Section 5: Conflicts / Surprises

**None identified.** The existing `generate_batch()` signature and the `Generator` Protocol are flexible enough to support a batching adapter without modification. The resume-on-restart mechanism (via `_load_existing()`) can coexist with batch persistence as long as batch IDs are tracked separately from the output JSONL file. Snapshot pinning is straightforward since the spec already carries the full model_id; no schema changes needed.
