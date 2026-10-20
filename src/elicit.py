"""Citation-elicitation generation runner.

Adapter pattern: each provider (Anthropic, OpenAI, Google, OpenAI-compatible,
local vLLM) implements an async `generate(prompt, spec)` method returning a
dict with output_text + token counts. The orchestrator (`generate_batch`) takes
questions × conditions and writes JSONL records, with resume-on-restart.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable, Iterator
from typing import Any, Literal, Protocol


Domain = Literal["academic", "medical", "legal"]
ConditionName = Literal[
    "citations_expected",
    "neutral",
    "abstention_permitted",
    "length_matched",
]

# Canonical separator for Anthropic Message Batches custom_id encoding.
# §11 sub-amendment 2026-06-07: the registered §14 long-form
# `{question_id}__{model_snapshot}__{condition}__t{trial}` exceeds Anthropic's
# 64-char Batch-API custom_id limit (75 chars on real Phase B inputs). The
# encoding is replaced with compact code tables below; the scientific
# semantics are unchanged because BatchKey still stores the full registered
# field values and round-trips losslessly through encode/decode. The code
# tables ARE the authoritative API-key audit trail.
_BATCH_KEY_SEP = "__"

# Compact code tables: each maps a registered field value to a short token
# that satisfies Anthropic's `^[a-zA-Z0-9_-]{1,64}$` custom_id pattern.
_DOMAIN_CODE = {"academic": "a", "medical": "m", "legal": "l"}
_CODE_DOMAIN = {v: k for k, v in _DOMAIN_CODE.items()}
_CELL_CODE = {"known_real": "k", "obscure_real": "o", "unanswerable": "u"}
_CODE_CELL = {v: k for k, v in _CELL_CODE.items()}
_CONDITION_CODE = {
    "neutral": "nu",
    "citations_expected": "ce",
    "abstention_permitted": "ap",
    "length_matched": "lm",
}
_CODE_CONDITION = {v: k for k, v in _CONDITION_CODE.items()}
_MODEL_CODE = {
    "claude-haiku-4-5-20251001": "h45d",
    "claude-opus-4-7": "o47",
    "claude-sonnet-4-6": "s46",
    # Phase C open-weights checkpoints (data/models/phase_c_checkpoints.jsonl).
    # The compact-code join-key mechanism is registered in §11 amendment #7; the
    # custom_id is a lossless join key, so extending the table to the 12 Phase C
    # model_ids preserves §7's semantics. Codes: <family><stage> (b=base, s=SFT,
    # d=DPO, r=RLVR, i=Instruct), each '/'-free and unique.
    "meta-llama/Llama-3.1-8B": "t8b",
    "allenai/Llama-3.1-Tulu-3-8B-SFT": "t8s",
    "allenai/Llama-3.1-Tulu-3-8B-DPO": "t8d",
    "allenai/Llama-3.1-Tulu-3-8B": "t8r",
    "allenai/OLMo-2-1124-13B": "o13b",
    "allenai/OLMo-2-1124-13B-SFT": "o13s",
    "allenai/OLMo-2-1124-13B-DPO": "o13d",
    "allenai/OLMo-2-1124-13B-Instruct": "o13i",
    "meta-llama/Llama-3.1-70B": "t70b",
    "allenai/Llama-3.1-Tulu-3-70B-SFT": "t70s",
    "allenai/Llama-3.1-Tulu-3-70B-DPO": "t70d",
    "allenai/Llama-3.1-Tulu-3-70B": "t70r",
    # Registered breadth-arm checkpoints collected 2026-09-06 under the
    # pre-committed rule in docs/sub_amendment_13_roster_reduction_draft.md.
    # Join keys only, same as the amendment #11 entries above; the base half of
    # the Llama pair is meta-llama/Llama-3.1-70B ("t70b").
    "meta-llama/Llama-3.1-70B-Instruct": "l70i",
    "Qwen/Qwen2.5-72B": "q72b",
    "Qwen/Qwen2.5-72B-Instruct": "q72i",
}
_CODE_MODEL = {v: k for k, v in _MODEL_CODE.items()}

import re as _re

_QID_ENCODE_RE = _re.compile(
    r"^(?P<domain>academic|medical|legal)"
    r"_(?P<cell>known_real|obscure_real|unanswerable)"
    r"_(?P<idx>\d{4})$"
)
_QID_DECODE_RE = _re.compile(r"^([aml])([kou])(\d{4})$")


def _qid_encode(qid: str) -> str:
    m = _QID_ENCODE_RE.match(qid)
    if m is None:
        raise ValueError(
            f"BatchKey.question_id does not match registered pattern "
            f"'<domain>_<cell>_<NNNN>': {qid!r}"
        )
    return f"{_DOMAIN_CODE[m['domain']]}{_CELL_CODE[m['cell']]}{m['idx']}"


def _qid_decode(short: str) -> str:
    m = _QID_DECODE_RE.match(short)
    if m is None:
        raise ValueError(
            f"Short question_id does not match '[aml][kou]NNNN': {short!r}"
        )
    return f"{_CODE_DOMAIN[m.group(1)]}_{_CODE_CELL[m.group(2)]}_{m.group(3)}"


@dataclass(frozen=True)
class BatchKey:
    """Round-trippable encoding for Anthropic Batch API custom_id.

    Joins back per-batch results to (question, model, condition, trial).
    Stores the full registered field values; encode() projects them into
    the compact code-table form to satisfy Anthropic's 64-char custom_id
    cap. decode() inverts via the same tables. Encoding fails fast on
    unregistered values so a drifted dataset/snapshot/condition surfaces
    before any paid batch submission.
    """

    question_id: str
    model_snapshot: str
    condition: str
    trial_id: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("question_id", self.question_id),
            ("model_snapshot", self.model_snapshot),
            ("condition", self.condition),
        ):
            if _BATCH_KEY_SEP in value:
                raise ValueError(
                    f"BatchKey.{field_name} cannot contain '{_BATCH_KEY_SEP}': {value!r}"
                )
        if self.trial_id < 0:
            raise ValueError(f"BatchKey.trial_id must be >= 0, got {self.trial_id}")

    def encode(self) -> str:
        q_short = _qid_encode(self.question_id)
        if self.model_snapshot not in _MODEL_CODE:
            raise ValueError(
                f"BatchKey.model_snapshot not registered in _MODEL_CODE: "
                f"{self.model_snapshot!r}"
            )
        if self.condition not in _CONDITION_CODE:
            raise ValueError(
                f"BatchKey.condition not registered in _CONDITION_CODE: "
                f"{self.condition!r}"
            )
        return (
            f"{q_short}{_BATCH_KEY_SEP}"
            f"{_MODEL_CODE[self.model_snapshot]}{_BATCH_KEY_SEP}"
            f"{_CONDITION_CODE[self.condition]}{_BATCH_KEY_SEP}"
            f"t{self.trial_id}"
        )

    @classmethod
    def decode(cls, custom_id: str) -> "BatchKey":
        parts = custom_id.split(_BATCH_KEY_SEP)
        if len(parts) != 4:
            raise ValueError(
                f"Expected 4 '{_BATCH_KEY_SEP}'-separated parts, got {len(parts)}: "
                f"{custom_id!r}"
            )
        q_short, m_short, c_short, trial_str = parts
        if not (trial_str.startswith("t") and trial_str[1:].isdigit()):
            raise ValueError(
                f"Trial component must match 't<digits>', got {trial_str!r}"
            )
        if m_short not in _CODE_MODEL:
            raise ValueError(f"Unknown model code in custom_id: {m_short!r}")
        if c_short not in _CODE_CONDITION:
            raise ValueError(f"Unknown condition code in custom_id: {c_short!r}")
        return cls(
            question_id=_qid_decode(q_short),
            model_snapshot=_CODE_MODEL[m_short],
            condition=_CODE_CONDITION[c_short],
            trial_id=int(trial_str[1:]),
        )


def parse_batch_results(results_iter: Iterable[Any]) -> Iterator[GenerationRecord]:
    """Convert per-batch results (MessageBatchIndividualResponse-like) into
    GenerationRecord objects.

    The `result.type` discriminator routes:
      - 'succeeded' → record with output_text + tokens + finish_reason
      - 'errored' / 'expired' / 'canceled' → record with `error` populated and
        empty output_text + None token counts

    Input objects are duck-typed: any object with `.custom_id` and `.result`
    works (tests use SimpleNamespace; production iterates
    `AsyncJSONLDecoder[MessageBatchIndividualResponse]` from the SDK).
    """
    for response in results_iter:
        key = BatchKey.decode(response.custom_id)
        result = response.result
        result_type = getattr(result, "type", None)
        if result_type == "succeeded":
            message = result.message
            text_blocks = [
                b.text for b in message.content if getattr(b, "type", None) == "text"
            ]
            yield GenerationRecord(
                question_id=key.question_id,
                condition=key.condition,
                model_id=key.model_snapshot,
                output_text="".join(text_blocks),
                input_tokens=message.usage.input_tokens,
                output_tokens=message.usage.output_tokens,
                finish_reason=getattr(message, "stop_reason", None),
                timestamp=_now_iso(),
                trial_id=key.trial_id,
            )
        else:
            error_obj = getattr(result, "error", None)
            if error_obj is not None:
                err_type = getattr(error_obj, "type", result_type or "unknown_error")
                err_msg = getattr(error_obj, "message", "")
                error_str = f"{err_type}: {err_msg}".rstrip(": ")
            else:
                error_str = str(result_type or "unknown_error")
            yield GenerationRecord(
                question_id=key.question_id,
                condition=key.condition,
                model_id=key.model_snapshot,
                output_text="",
                input_tokens=None,
                output_tokens=None,
                finish_reason=None,
                timestamp=_now_iso(),
                trial_id=key.trial_id,
                error=error_str,
            )


def build_batch_requests(
    questions: list["Question"],
    conditions: list["PromptCondition"],
    spec: "GenerationSpec",
    num_trials: int = 1,
) -> list[dict[str, Any]]:
    """Construct Anthropic Message Batches Request payloads.

    Returns plain dicts compatible with the SDK's
    `Iterable[batch_create_params.Request]` parameter. Each request carries
    a custom_id round-trippable via BatchKey and params matching the shape
    used by the sync AnthropicGenerator (model, max_tokens, temperature,
    messages).
    """
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}")

    requests: list[dict[str, Any]] = []
    for question in questions:
        for condition in conditions:
            prompt = condition.render(question)
            for trial_id in range(num_trials):
                key = BatchKey(
                    question_id=question.id,
                    model_snapshot=spec.model_id,
                    condition=str(condition.name),
                    trial_id=trial_id,
                )
                requests.append(
                    {
                        "custom_id": key.encode(),
                        "params": {
                            "model": spec.model_id,
                            "max_tokens": spec.max_output_tokens,
                            "temperature": spec.temperature,
                            "messages": [{"role": "user", "content": prompt}],
                        },
                    }
                )
    return requests


@dataclass(frozen=True)
class Question:
    id: str
    domain: Domain
    text: str
    expected_existence: Literal["answerable", "unanswerable"]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptCondition:
    name: ConditionName | str
    template: str  # str.format() template with {question} placeholder

    def render(self, question: Question) -> str:
        return self.template.format(question=question.text)


@dataclass(frozen=True)
class GenerationSpec:
    model_id: str
    provider: Literal["anthropic", "openai", "google", "openai_compatible", "vllm_local", "fake"]
    temperature: float = 1.0
    max_output_tokens: int = 1024
    api_base: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationRecord:
    question_id: str
    condition: str
    model_id: str
    output_text: str
    input_tokens: int | None
    output_tokens: int | None
    finish_reason: str | None
    timestamp: str
    trial_id: int = 0
    seed: int | None = None
    error: str | None = None

    def to_jsonl_dict(self) -> dict[str, Any]:
        return asdict(self)


class Generator(Protocol):
    async def generate(self, prompt: str, spec: GenerationSpec) -> dict[str, Any]:
        """Return dict with keys: output_text, input_tokens, output_tokens, finish_reason."""
        ...


# ---------------- Anthropic adapter ----------------

class AnthropicGenerator:
    """Adapter for the Anthropic Messages API.

    Pass either an instantiated AsyncAnthropic client or let the constructor
    create one (requires ANTHROPIC_API_KEY in env).
    """

    def __init__(self, client: Any | None = None):
        if client is None:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic()
        self._client = client

    async def generate(self, prompt: str, spec: GenerationSpec) -> dict[str, Any]:
        response = await self._client.messages.create(
            model=spec.model_id,
            max_tokens=spec.max_output_tokens,
            temperature=spec.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        return {
            "output_text": "".join(text_blocks),
            "input_tokens": getattr(response.usage, "input_tokens", None),
            "output_tokens": getattr(response.usage, "output_tokens", None),
            "finish_reason": getattr(response, "stop_reason", None),
        }


# ---------------- Local vLLM adapter (Phase C open-weights) ----------------

def _vllm_sampling_kwargs(spec: GenerationSpec) -> dict[str, Any]:
    """Project a GenerationSpec onto vllm.SamplingParams constructor kwargs.

    Pure (no vLLM import) so it is testable without a GPU. `seed` is included
    only when present in spec.extra; open-weights runs pin temperature=0 and a
    documented seed for reproducibility (pre-reg §6 trial design).
    """
    kwargs: dict[str, Any] = {
        "temperature": spec.temperature,
        "max_tokens": spec.max_output_tokens,
    }
    if "seed" in spec.extra:
        kwargs["seed"] = spec.extra["seed"]
    return kwargs


def _parse_vllm_output(request_output: Any) -> dict[str, Any]:
    """Parse one vllm.RequestOutput into the Generator result dict.

    Token counts come from the tokenized prompt/completion id lists, which vLLM
    populates directly, so they are exact rather than re-estimated.
    """
    completion = request_output.outputs[0]
    return {
        "output_text": completion.text,
        "input_tokens": len(request_output.prompt_token_ids),
        "output_tokens": len(completion.token_ids),
        "finish_reason": completion.finish_reason,
    }


class VLLMGenerator:
    """Adapter for local open-weights inference via vLLM offline batching.

    Inference engine and SamplingParams class are injectable so the logic is
    testable without a GPU; production lazy-imports the real classes when they
    are left as None.

    Chat-template branch (scientifically load-bearing): base checkpoints have no
    chat template and must use raw completion (`llm.generate`); post-trained
    checkpoints (SFT/DPO/RLVR/Instruct) use `llm.chat`. The per-checkpoint
    choice is set by `apply_chat_template` (or overridden per-spec via
    spec.extra["apply_chat_template"]) so a base model is never fed a chat
    template, which would corrupt the base-vs-post-trained contrast E1 measures.
    """

    def __init__(
        self,
        llm: Any | None = None,
        model_id: str | None = None,
        apply_chat_template: bool = True,
        sampling_params_cls: Any | None = None,
        engine_kwargs: dict[str, Any] | None = None,
    ):
        if llm is None:
            from vllm import LLM
            llm = LLM(model=model_id, **(engine_kwargs or {}))
        if sampling_params_cls is None:
            from vllm import SamplingParams
            sampling_params_cls = SamplingParams
        self._llm = llm
        self._apply_chat_template = apply_chat_template
        self._sampling_params_cls = sampling_params_cls

    def _use_chat(self, spec: GenerationSpec) -> bool:
        return bool(spec.extra.get("apply_chat_template", self._apply_chat_template))

    def generate_many(self, prompts: list[str], spec: GenerationSpec) -> list[dict[str, Any]]:
        """Batched generation over all prompts in one vLLM call (the GPU-sweep path)."""
        sampling_params = self._sampling_params_cls(**_vllm_sampling_kwargs(spec))
        if self._use_chat(spec):
            conversations = [[{"role": "user", "content": p}] for p in prompts]
            outputs = self._llm.chat(conversations, sampling_params)
        else:
            outputs = self._llm.generate(prompts, sampling_params)
        return [_parse_vllm_output(o) for o in outputs]

    async def generate(self, prompt: str, spec: GenerationSpec) -> dict[str, Any]:
        """Protocol-compatible single-prompt path (dev/compose with generate_batch)."""
        return self.generate_many([prompt], spec)[0]


# ---------------- Batch orchestrator ----------------

def _record_key(question_id: str, condition: str, model_id: str) -> tuple[str, str, str]:
    return (question_id, condition, model_id)


def _load_existing(output_path: Path) -> set[tuple[str, str, str]]:
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str, str]] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        keys.add(_record_key(record["question_id"], record["condition"], record["model_id"]))
    return keys


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def generate_offline_batch(
    questions: list[Question],
    conditions: list[PromptCondition],
    spec: GenerationSpec,
    generator: Any,
    output_path: Path,
    trial_id: int = 0,
) -> None:
    """Synchronous offline-batch generation for local vLLM (Phase C GPU sweep).

    Unlike `generate_batch` (async, one network call per prompt), this renders
    every outstanding (question x condition) prompt and hands the whole list to
    `generator.generate_many` in a single call so vLLM's continuous batching is
    actually exercised. Open-weights runs are temp=0, 1 trial (pre-reg §6), so
    there is no trial loop here.

    Idempotent: any (question_id, condition, model_id) already in the output is
    skipped, so a sweep that dies mid-checkpoint resumes without re-billing GPU
    hours.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(output_path)

    pending: list[tuple[Question, PromptCondition]] = []
    for question in questions:
        for condition in conditions:
            key = _record_key(question.id, condition.name, spec.model_id)
            if key not in existing:
                pending.append((question, condition))

    if not pending:
        return

    prompts = [condition.render(question) for question, condition in pending]
    results = generator.generate_many(prompts, spec)

    with output_path.open("a", encoding="utf-8") as handle:
        for (question, condition), result in zip(pending, results):
            record = GenerationRecord(
                question_id=question.id,
                condition=condition.name,
                model_id=spec.model_id,
                output_text=result["output_text"],
                input_tokens=result.get("input_tokens"),
                output_tokens=result.get("output_tokens"),
                finish_reason=result.get("finish_reason"),
                timestamp=_now_iso(),
                trial_id=trial_id,
                seed=spec.extra.get("seed"),
            )
            handle.write(json.dumps(record.to_jsonl_dict()) + "\n")


async def generate_batch(
    questions: list[Question],
    conditions: list[PromptCondition],
    spec: GenerationSpec,
    generator: Generator,
    output_path: Path,
    concurrency: int = 4,
) -> None:
    """Run (question × condition) generations and append JSONL records.

    Idempotent: any (question_id, condition, model_id) already present in the
    output file is skipped, so killing the process and restarting picks up
    where it left off.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(output_path)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async def _one(question: Question, condition: PromptCondition) -> None:
        key = _record_key(question.id, condition.name, spec.model_id)
        if key in existing:
            return
        prompt = condition.render(question)
        async with semaphore:
            try:
                result = await generator.generate(prompt, spec)
                record = GenerationRecord(
                    question_id=question.id,
                    condition=condition.name,
                    model_id=spec.model_id,
                    output_text=result["output_text"],
                    input_tokens=result.get("input_tokens"),
                    output_tokens=result.get("output_tokens"),
                    finish_reason=result.get("finish_reason"),
                    timestamp=_now_iso(),
                )
            except Exception as exc:
                record = GenerationRecord(
                    question_id=question.id,
                    condition=condition.name,
                    model_id=spec.model_id,
                    output_text="",
                    input_tokens=None,
                    output_tokens=None,
                    finish_reason=None,
                    timestamp=_now_iso(),
                    error=f"{type(exc).__name__}: {exc}",
                )
        async with write_lock:
            with output_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_jsonl_dict()) + "\n")

    tasks = [_one(q, c) for q in questions for c in conditions]
    await asyncio.gather(*tasks)


# ---------------- Batch API: manifest sidecar ----------------


@dataclass(frozen=True)
class BatchManifestEntry:
    """One row in the batch-submission ledger.

    Append-only: the orchestrator writes a row on submission and another row
    when the batch ends. `load_manifest` reduces by `batch_id` keeping the
    most recent entry, so a partially-written manifest never confuses the
    resume logic.
    """

    batch_id: str
    model_id: str
    output_path: str
    request_count: int
    submitted_at: str
    ended_at: str | None = None
    final_status: str | None = None

    def to_jsonl_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_jsonl_dict(cls, d: dict[str, Any]) -> "BatchManifestEntry":
        return cls(**d)


def write_manifest_entry(manifest_path: Path, entry: BatchManifestEntry) -> None:
    """Append one entry to the manifest JSONL (creating the file if needed)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry.to_jsonl_dict()) + "\n")


def load_manifest(manifest_path: Path) -> list[BatchManifestEntry]:
    """Read the manifest and collapse to one entry per batch_id (latest wins)."""
    if not manifest_path.exists():
        return []
    latest_by_id: dict[str, BatchManifestEntry] = {}
    order: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        entry = BatchManifestEntry.from_jsonl_dict(json.loads(line))
        if entry.batch_id not in latest_by_id:
            order.append(entry.batch_id)
        latest_by_id[entry.batch_id] = entry
    return [latest_by_id[bid] for bid in order]


# ---------------- Batch API: submit / poll / fetch ----------------


def _load_existing_keys_with_trial(
    output_path: Path,
) -> set[tuple[str, str, str, int]]:
    if not output_path.exists():
        return set()
    keys: set[tuple[str, str, str, int]] = set()
    for line in output_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        keys.add(
            (
                record["question_id"],
                record["condition"],
                record["model_id"],
                int(record.get("trial_id", 0)),
            )
        )
    return keys


async def submit_batch(
    *,
    client: Any,
    questions: list[Question],
    conditions: list[PromptCondition],
    spec: GenerationSpec,
    num_trials: int,
    manifest_path: Path,
    output_path: Path,
) -> BatchManifestEntry:
    """Build the request payload, post to /messages/batches, write the manifest entry."""
    requests = build_batch_requests(questions, conditions, spec, num_trials=num_trials)
    if not requests:
        raise ValueError(
            "submit_batch produced zero requests — questions or conditions list is empty"
        )
    response = await client.messages.batches.create(requests=requests)
    entry = BatchManifestEntry(
        batch_id=response.id,
        model_id=spec.model_id,
        output_path=str(output_path),
        request_count=len(requests),
        submitted_at=_now_iso(),
    )
    write_manifest_entry(manifest_path, entry)
    return entry


async def poll_until_ended(
    *,
    client: Any,
    batch_id: str,
    poll_interval_s: float = 30.0,
    timeout_s: float = 25 * 3600,
) -> Any:
    """Poll messages.batches.retrieve until processing_status == 'ended'.

    Raises TimeoutError if the batch is still in flight after `timeout_s`.
    `poll_interval_s` is the gap between retrieve() calls; the actual SLA
    cap of 24h justifies the 25h default timeout.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        status_obj = await client.messages.batches.retrieve(batch_id)
        if status_obj.processing_status == "ended":
            return status_obj
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Batch {batch_id} did not end within {timeout_s}s "
                f"(last status: {status_obj.processing_status})"
            )
        await asyncio.sleep(poll_interval_s)


async def fetch_and_write_results(
    *,
    client: Any,
    batch_id: str,
    output_path: Path,
) -> None:
    """Stream results from /messages/batches/{id}/results into the output JSONL.

    Idempotent: any (question_id, condition, model_id, trial_id) already
    present in the output is skipped, so re-running after a partial fetch
    or process crash does not duplicate rows.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_keys_with_trial(output_path)
    iterator = await client.messages.batches.results(batch_id)
    response_buffer: list[Any] = []
    async for item in iterator:
        response_buffer.append(item)
    with output_path.open("a", encoding="utf-8") as fh:
        for record in parse_batch_results(response_buffer):
            key = (
                record.question_id,
                record.condition,
                record.model_id,
                record.trial_id,
            )
            if key in existing:
                continue
            fh.write(json.dumps(record.to_jsonl_dict()) + "\n")
            existing.add(key)


# ---------------- Batch API: top-level orchestrator ----------------


def _expected_keys(
    questions: list[Question],
    conditions: list[PromptCondition],
    spec: GenerationSpec,
    num_trials: int,
) -> set[tuple[str, str, str, int]]:
    return {
        (q.id, str(c.name), spec.model_id, t)
        for q in questions
        for c in conditions
        for t in range(num_trials)
    }


def _find_in_progress_entry(
    manifest_path: Path, model_id: str, output_path: Path
) -> BatchManifestEntry | None:
    output_str = str(output_path)
    found: BatchManifestEntry | None = None
    for entry in load_manifest(manifest_path):
        if (
            entry.model_id == model_id
            and entry.output_path == output_str
            and entry.final_status is None
        ):
            found = entry  # keep scanning; latest in-file wins
    return found


async def generate_via_batch_api(
    *,
    questions: list[Question],
    conditions: list[PromptCondition],
    spec: GenerationSpec,
    client: Any,
    output_path: Path,
    manifest_path: Path,
    num_trials: int = 1,
    poll_interval_s: float = 30.0,
    poll_timeout_s: float = 25 * 3600,
) -> None:
    """End-to-end batch run with resume-on-restart.

    Behavior:
      1. If the manifest has an in-progress batch for (model_id, output_path),
         resume it (poll + fetch). Don't submit a new one.
      2. Otherwise, if every expected (q, c, trial) is already in the output,
         return without contacting the API.
      3. Otherwise, submit a new batch with all requested work, poll to
         completion, fetch results, append to output JSONL.
      4. Write a closing manifest row with final_status from the API response.
    """
    in_progress = _find_in_progress_entry(manifest_path, spec.model_id, output_path)

    if in_progress is None:
        expected = _expected_keys(questions, conditions, spec, num_trials)
        existing = _load_existing_keys_with_trial(output_path)
        if expected <= existing:
            return
        entry = await submit_batch(
            client=client,
            questions=questions,
            conditions=conditions,
            spec=spec,
            num_trials=num_trials,
            manifest_path=manifest_path,
            output_path=output_path,
        )
        batch_id = entry.batch_id
        request_count = entry.request_count
        submitted_at = entry.submitted_at
    else:
        batch_id = in_progress.batch_id
        request_count = in_progress.request_count
        submitted_at = in_progress.submitted_at

    final = await poll_until_ended(
        client=client,
        batch_id=batch_id,
        poll_interval_s=poll_interval_s,
        timeout_s=poll_timeout_s,
    )
    await fetch_and_write_results(
        client=client, batch_id=batch_id, output_path=output_path
    )
    write_manifest_entry(
        manifest_path,
        BatchManifestEntry(
            batch_id=batch_id,
            model_id=spec.model_id,
            output_path=str(output_path),
            request_count=request_count,
            submitted_at=submitted_at,
            ended_at=_now_iso(),
            final_status=getattr(final, "processing_status", "ended"),
        ),
    )
