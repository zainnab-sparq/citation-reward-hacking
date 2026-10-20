"""E4's higher-capacity arm: DeBERTa-v3-base over citation spans only.

The model reads the concatenated raw citation spans and nothing else, so it is
blind to existence and to the answer's semantics while still being higher
capacity than the 4-feature logistic. That is what makes the registered 5pp
ablation informative: taking §5's "only surface features" literally would put a
transformer over four scalars, which is a strictly worse logistic regression.

Hyperparameters are fixed here rather than tuned. There is no validation split
and no early stopping, deliberately: E4 is a one-shot registered test, and a
tuning loop over a held-out slice of the training family would be a search for a
preference rate. The settings are ordinary DeBERTa fine-tuning defaults.

Imports of torch/transformers are deferred into the functions so the rest of the
E4 analysis, and the test suite, run without the ML extras installed.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 256
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
EPOCHS = 3
WARMUP_FRACTION = 0.1
SEED = 20260813
INFERENCE_BATCH_SIZE = 64
MAX_GRAD_NORM = 1.0


class TrainingDivergedError(RuntimeError):
    """Raised the first time the loss goes non-finite, rather than at the end."""


def require_finite_loss(value: float, step: int) -> None:
    """Fail on the first non-finite loss instead of training to completion.

    The first attempt at this ran 200 steps at `mean_loss nan` and would have
    spent fifteen minutes producing a model of pure noise before anyone looked.
    A silent NaN is worse than a crash here: the run still writes a preference
    rate, and it looks like a result.
    """
    if not math.isfinite(value):
        raise TrainingDivergedError(
            f"loss became {value} at step {step}. Check that the model was loaded in float32: "
            "transformers 5.x keeps the checkpoint's stored dtype, and DeBERTa-v3 ships fp16, "
            "which diverges under AdamW with no gradient scaler."
        )


def batched(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    if size < 1:
        raise ValueError(f"batch size must be positive, got {size}")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _resolve_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_tokenizer():
    """DeBERTa-v3 ships a sentencepiece vocab that transformers 5.x converts
    through tiktoken, so `tiktoken` and `protobuf` are hard requirements here
    even though nothing in this project uses either directly. `use_fast=False`
    does NOT route around it on 5.x; the conversion runs either way."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_NAME)


class DebertaScorer:
    """Wraps a fine-tuned model as `texts -> P(post-trained)`."""

    def __init__(self, model, tokenizer, device: str):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def __call__(self, texts: Sequence[str]) -> list[float]:
        import torch

        self._model.eval()
        scores: list[float] = []
        with torch.no_grad():
            for chunk in batched(list(texts), INFERENCE_BATCH_SIZE):
                encoded = self._tokenizer(
                    list(chunk), truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
                ).to(self._device)
                logits = self._model(**encoded).logits
                scores.extend(torch.softmax(logits, dim=-1)[:, 1].tolist())
        return scores


def fit_deberta(texts: Sequence[str], labels: Sequence[int], *, log=print) -> DebertaScorer:
    """Fine-tune the binary post-trained/base classifier on the training family."""
    import torch
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    if len(texts) != len(labels):
        raise ValueError(f"{len(texts)} texts against {len(labels)} labels")

    torch.manual_seed(SEED)
    device = _resolve_device()
    tokenizer = _load_tokenizer()
    # float32 is REQUIRED, not a preference. transformers 5.x loads a checkpoint
    # in its stored dtype instead of upcasting, and DeBERTa-v3-base is stored in
    # fp16; training it in pure fp16 under AdamW with no gradient scaler drives
    # the loss to NaN within a couple of hundred steps while the forward pass
    # still looks finite at initialisation.
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, dtype=torch.float32
    ).to(device)

    indices = list(range(len(texts)))
    steps_per_epoch = (len(indices) + BATCH_SIZE - 1) // BATCH_SIZE
    total_steps = steps_per_epoch * EPOCHS
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRACTION * total_steps), total_steps)

    generator = torch.Generator().manual_seed(SEED)
    model.train()
    for epoch in range(EPOCHS):
        order = torch.randperm(len(indices), generator=generator).tolist()
        running = 0.0
        for step, chunk in enumerate(batched(order, BATCH_SIZE), start=1):
            encoded = tokenizer(
                [texts[i] for i in chunk], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
            ).to(device)
            target = torch.tensor([labels[i] for i in chunk], device=device)
            loss = model(**encoded, labels=target).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            require_finite_loss(loss.item(), step)
            running += loss.item()
            if step % 200 == 0:
                log(f"  epoch {epoch + 1}/{EPOCHS} step {step}/{steps_per_epoch} mean_loss {running / step:.4f}")
        log(f"  epoch {epoch + 1} done, mean_loss {running / steps_per_epoch:.4f}")

    return DebertaScorer(model, tokenizer, device)
