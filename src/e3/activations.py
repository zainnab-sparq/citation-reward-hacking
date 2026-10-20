"""Extract answer-token hidden states for the E3 probe.

H1.3 is a claim about hidden states "at the answer-token positions", so the
answer span is found by tokenizing the prompt alone and the prompt-plus-answer,
and taking the difference. Doing it that way rather than by string search means
the boundary is exact even when the tokenizer merges the last prompt character
with the first answer character.

The model is loaded 4-bit because the registered primary target is a 70B and
the machine available for the registered *scale-check* target (Tulu-3-8B) has
12 GB of VRAM, which does not hold an 8B model in bf16. Quantization changes
the activations being probed and is disclosed as such; it is a property of the
scale-check arm, not of the registered primary.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

MODEL_ID = "allenai/Llama-3.1-Tulu-3-8B"
BASE_MODEL_ID = "meta-llama/Llama-3.1-8B"


def answer_token_span(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> tuple[int, int]:
    """Half-open [start, end) token range of the answer inside `full_ids`."""
    prompt_ids, full_ids = list(prompt_ids), list(full_ids)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "full sequence does not begin with the prompt; the answer span "
            "cannot be located by prefix difference"
        )
    if len(full_ids) <= len(prompt_ids):
        raise ValueError("no answer tokens: full sequence is not longer than the prompt")
    return (len(prompt_ids), len(full_ids))


def mean_pool_span(hidden: np.ndarray, span: tuple[int, int]) -> np.ndarray:
    """Average hidden states over the answer span only."""
    start, end = span
    hidden = np.asarray(hidden)
    if start < 0 or end > hidden.shape[0] or start >= end:
        raise ValueError(
            f"span {span} is out of range for {hidden.shape[0]} token positions"
        )
    return hidden[start:end].mean(axis=0)


def pool_all_layers(layers: Sequence[np.ndarray], span: tuple[int, int]) -> np.ndarray:
    """One pooled vector per layer, layer order preserved.

    Order matters: the layer index is the unit of the registered per-layer
    ablation, so silently reordering here would relabel the whole sweep.
    """
    return np.stack([mean_pool_span(layer, span) for layer in layers])


def load_model(model_id: str = MODEL_ID, four_bit: bool = True):
    """Load the probe target. Kept out of the pure functions above so the
    extraction logic stays testable without a GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto", "output_hidden_states": True}
    if four_bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def extract_one(model, tokenizer, prompt: str, answer: str) -> np.ndarray:
    """Pooled answer-token hidden states for one (prompt, answer) pair.

    Returns an array of shape (n_layers, hidden_size).
    """
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tokenizer(prompt + answer, add_special_tokens=True)["input_ids"]
    span = answer_token_span(prompt_ids, full_ids)

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor([full_ids], device=model.device),
            output_hidden_states=True,
        )
    layers = [h[0].float().cpu().numpy() for h in out.hidden_states]
    return pool_all_layers(layers, span)
