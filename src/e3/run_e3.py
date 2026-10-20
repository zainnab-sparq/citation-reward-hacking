"""Run E3: train the truthfulness probe, apply it to citation activations.

Order of operations matters here and is fixed in advance.

The headline layer is chosen on the PROBE set's own validation split, never on
the citation activations. Picking the layer that maximizes citation AUROC would
be selecting on the outcome, and with 33 layers that maximum is a maximum over
33 correlated tries rather than an estimate. The full per-layer sweep is
reported regardless, which is the registered ablation.

The within-domain probe trains on one subset of citations and evaluates on a
disjoint one, split by question so no question straddles the boundary. Its
AUROC feeds the pre-committed agreement rule against the cross-task probe.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np

from src.e3.activations import extract_one, load_model
from src.e3.citation_spans import char_span_to_token_span
from src.e3.difficulty_proxy import load_difficulty_proxy
from src.e3.probe import (
    apply_agreement_rule,
    difficulty_deconfound_verdict,
    difficulty_probe,
    fit_probe,
    meets_h1_3,
    proxy_alone_auroc,
    shuffled_label_control,
    sweep_layers,
)
from src.e3.probe_dataset import (
    build_probe_dataset,
    load_source_datasets,
    split_probe_dataset,
)

CITATION_ROWS = Path("data/activations/e3_citation_rows.jsonl")
GENERATIONS = Path("data/generations/phase_c")
QUESTIONS = Path("data/questions/frozen_900q_legal_amended.jsonl")
PROBE_MODEL = "allenai/Llama-3.1-Tulu-3-8B"
OUT_DIR = Path("data/analysis/phase_c")
SEED = 42

# Pre-registration section 6, guardrail (a): "the most interpretable positive
# result is the fabrication probe out-performing the difficulty proxy
# specifically on the high-difficulty obscure-but-real + unanswerable items."
HIGH_DIFFICULTY_CELLS = ("obscure_real", "unanswerable")


def load_condition_templates() -> dict[str, str]:
    """The exact templates Phase C generated under.

    Imported rather than retyped: the probe reads hidden states in the context
    the answer was actually produced in, so a prompt that differs from the
    generation-time prompt would probe a state the model never occupied.
    """
    from src.pilot import CONDITIONS

    return {str(c.name): c.template for c in CONDITIONS}


def load_question_text() -> dict[str, str]:
    return {q["id"]: q["question"] for q in _read_jsonl(QUESTIONS)}


def load_question_cells() -> dict[str, str]:
    return {q["id"]: q["cell"] for q in _read_jsonl(QUESTIONS)}


def _format_prompt(tokenizer, question: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def extract_probe_activations(model, tokenizer, rows, log_every=500):
    vecs, labels = [], []
    t0 = time.time()
    for i, row in enumerate(rows):
        prompt = _format_prompt(tokenizer, row.question)
        vecs.append(extract_one(model, tokenizer, prompt, row.answer))
        labels.append(row.label)
        if log_every and (i + 1) % log_every == 0:
            print(f"    probe {i + 1}/{len(rows)}  ({time.time() - t0:.0f}s)", flush=True)
    return np.stack(vecs), np.asarray(labels)


def extract_citation_activations(model, tokenizer, log_every=250, limit_gens=None):
    """One forward pass per generation, pooling every citation span inside it."""
    import torch

    from src.e3.activations import pool_all_layers

    rows = list(_read_jsonl(CITATION_ROWS))
    by_gen: dict[tuple, list[dict]] = {}
    for row in rows:
        by_gen.setdefault((row["question_id"], row["condition"], row["trial_id"]), []).append(row)

    gen_path = GENERATIONS / (PROBE_MODEL.replace("/", "__") + ".jsonl")
    gens = {
        (g["question_id"], g["condition"], g.get("trial_id", 0)): g
        for g in _read_jsonl(gen_path)
    }

    templates = load_condition_templates()
    question_text = load_question_text()

    vecs, labels, questions, skipped = [], [], [], 0
    t0 = time.time()
    items = sorted(by_gen.items())
    if limit_gens is not None:
        items = items[:limit_gens]
    for i, (key, citations) in enumerate(items):
        gen = gens[key]
        answer = gen["output_text"]
        question_id, condition = key[0], key[1]
        rendered = templates[condition].format(question=question_text[question_id])
        prompt = _format_prompt(tokenizer, rendered)
        # Offsets are taken within the answer alone, then shifted by the prompt
        # length, so a character span in output_text maps to the right absolute
        # token position in the prompt-plus-answer sequence.
        enc = tokenizer(answer, add_special_tokens=False, return_offsets_mapping=True)
        answer_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = prompt_ids + answer_ids

        with torch.no_grad():
            out = model(
                input_ids=torch.tensor([full_ids], device=model.device),
                output_hidden_states=True,
            )
        layers = [h[0].float().cpu().numpy() for h in out.hidden_states]

        for row in citations:
            try:
                tok_span = char_span_to_token_span(offsets, (row["char_start"], row["char_end"]))
            except ValueError:
                skipped += 1
                continue
            shifted = (tok_span[0] + len(prompt_ids), tok_span[1] + len(prompt_ids))
            vecs.append(pool_all_layers(layers, shifted))
            labels.append(row["label"])
            questions.append(row["question_id"])

        if log_every and (i + 1) % log_every == 0:
            print(f"    citations {i + 1}/{len(items)} gens, "
                  f"{len(vecs)} vectors ({time.time() - t0:.0f}s)", flush=True)

    print(f"    citation spans skipped in token space: {skipped}")
    return np.stack(vecs), np.asarray(labels), questions


def split_by_question(questions, seed=SEED, frac=0.5):
    uniq = sorted(set(questions))
    random.Random(seed).shuffle(uniq)
    cut = int(len(uniq) * frac)
    train_q = set(uniq[:cut])
    idx_train = [i for i, q in enumerate(questions) if q in train_q]
    idx_test = [i for i, q in enumerate(questions) if q not in train_q]
    return idx_train, idx_test


def _verdict_dict(fabrication_auroc: float, difficulty_auroc: float) -> dict:
    v = difficulty_deconfound_verdict(fabrication_auroc, difficulty_auroc)
    return {
        "reads_difficulty": v.reads_difficulty,
        "reported_as": v.reported_as,
        "margin": v.margin,
    }


def _probe_pair(X_tr, y_tr, d_tr, X_te, y_te, d_te) -> dict:
    """Fabrication probe vs difficulty probe on the same held-out rows."""
    fabrication = fit_probe(X_tr, y_tr, X_te, y_te)
    difficulty = difficulty_probe(X_tr, d_tr, X_te, y_te)
    return {
        "n_test": int(len(y_te)),
        "fabrication_auroc": fabrication.auroc,
        "difficulty_probe_auroc": difficulty.auroc,
        "proxy_alone_auroc": proxy_alone_auroc(d_te, y_te),
        "verdict": _verdict_dict(fabrication.auroc, difficulty.auroc),
    }


def difficulty_deconfound_block(X, y, questions, proxy, cells, seed=SEED) -> dict:
    """Guardrail (a) on one layer's citation activations.

    The split by question is the within-domain probe's own split (same seed),
    so the fabrication AUROC here is the headline within-domain figure and the
    difficulty probe is scored on exactly the rows that produced it. Every
    question must carry a proxy value; a missing one is a join bug, not a
    question to fill with a default.
    """
    missing = sorted({q for q in questions if q not in proxy})
    if missing:
        raise ValueError(f"{len(missing)} question(s) have no difficulty proxy: {missing[:5]}")
    X, y = np.asarray(X), np.asarray(y)
    d = np.asarray([proxy[q] for q in questions], dtype=float)
    idx_tr, idx_te = split_by_question(questions, seed=seed)

    block = _probe_pair(X[idx_tr], y[idx_tr], d[idx_tr], X[idx_te], y[idx_te], d[idx_te])
    block["n_train"] = int(len(idx_tr))
    block["proxy_n_questions"] = int(len(set(questions)))

    hard = [i for i in idx_te if cells[questions[i]] in HIGH_DIFFICULTY_CELLS]
    subset = _probe_pair(X[idx_tr], y[idx_tr], d[idx_tr], X[hard], y[hard], d[hard])
    subset["cells"] = sorted(HIGH_DIFFICULTY_CELLS)
    block["high_difficulty_subset"] = subset
    return block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-probe", type=int, default=None,
                        help="cap source rows per dataset (smoke runs)")
    parser.add_argument("--limit-gens", type=int, default=None,
                        help="cap generations used for citation activations (smoke runs)")
    parser.add_argument("--out", default=str(OUT_DIR / "e3_results.json"))
    parser.add_argument("--cache", default="data/activations/e3_activations.npz",
                        help="where extracted activations are saved / loaded from")
    parser.add_argument("--from-cache", action="store_true",
                        help="skip the model entirely and load activations from --cache")
    parser.add_argument("--shuffle-seeds", type=int, default=1,
                        help="number of label permutations for the shuffled-label null band")
    parser.add_argument("--difficulty-proxy", default=None,
                        help="per-question proxy JSONL from src.e3.difficulty_proxy; "
                             "runs the registered difficulty deconfound (guardrail (a)) "
                             "only when given")
    args = parser.parse_args()

    cache = Path(args.cache)
    if args.from_cache:
        if not cache.exists():
            raise SystemExit(f"--from-cache given but {cache} does not exist")
        print(f"loading activations from {cache} ...", flush=True)
        z = np.load(cache, allow_pickle=True)
        X_train, y_train = z["X_train"], z["y_train"]
        X_val, y_val = z["X_val"], z["y_val"]
        X_cit, y_cit = z["X_cit"], z["y_cit"]
        cit_questions = list(z["cit_questions"])
    else:
        print("loading probe sources ...", flush=True)
        tqa, sciq = load_source_datasets(limit_per_source=args.limit_probe)
        rows = build_probe_dataset(tqa, sciq)
        train_rows, val_rows, _ = split_probe_dataset(rows, seed=SEED)
        print(f"  probe rows: {len(rows)} (train {len(train_rows)}, val {len(val_rows)})")

        print("loading model 4-bit ...", flush=True)
        model, tokenizer = load_model(four_bit=True)

        print("extracting probe activations ...", flush=True)
        X_train, y_train = extract_probe_activations(model, tokenizer, train_rows)
        X_val, y_val = extract_probe_activations(model, tokenizer, val_rows)

        print("extracting citation activations ...", flush=True)
        X_cit, y_cit, cit_questions = extract_citation_activations(
            model, tokenizer, limit_gens=args.limit_gens)

        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X_train=X_train, y_train=y_train, X_val=X_val,
                            y_val=y_val, X_cit=X_cit, y_cit=y_cit,
                            cit_questions=np.array(cit_questions))
        print(f"  activations cached to {cache}", flush=True)

    print(f"  citation vectors: {X_cit.shape}, "
          f"{int(y_cit.sum())} fabricated / {int((1 - y_cit).sum())} real")

    n_layers = X_train.shape[1]

    # Layer choice is made on the probe validation split only.
    val_sweep = sweep_layers(
        {l: (X_train[:, l, :], y_train) for l in range(n_layers)},
        {l: (X_val[:, l, :], y_val) for l in range(n_layers)},
    )
    best_layer = max(val_sweep, key=lambda l: val_sweep[l].auroc)
    print(f"  layer selected on probe validation: {best_layer} "
          f"(val AUROC {val_sweep[best_layer].auroc:.4f})")

    cross_task = fit_probe(X_train[:, best_layer, :], y_train,
                           X_cit[:, best_layer, :], y_cit)
    # One permutation is a single draw from the null, not the null. With several
    # seeds the band shows whether the observed cross-task AUROC sits inside what
    # a label-blind probe produces on these activations, which is the actual
    # leak question.
    shuffled_aurocs = [
        shuffled_label_control(X_train[:, best_layer, :], y_train,
                               X_cit[:, best_layer, :], y_cit, seed=SEED + k).auroc
        for k in range(max(1, args.shuffle_seeds))
    ]
    shuffled = shuffled_label_control(X_train[:, best_layer, :], y_train,
                                      X_cit[:, best_layer, :], y_cit, seed=SEED)

    citation_sweep = sweep_layers(
        {l: (X_train[:, l, :], y_train) for l in range(n_layers)},
        {l: (X_cit[:, l, :], y_cit) for l in range(n_layers)},
    )

    idx_tr, idx_te = split_by_question(cit_questions, seed=SEED)
    within = fit_probe(X_cit[idx_tr][:, best_layer, :], y_cit[idx_tr],
                       X_cit[idx_te][:, best_layer, :], y_cit[idx_te])

    verdict = apply_agreement_rule(cross_task.auroc, within.auroc)

    deconfound = None
    if args.difficulty_proxy:
        deconfound = difficulty_deconfound_block(
            X_cit[:, best_layer, :], y_cit, cit_questions,
            load_difficulty_proxy(Path(args.difficulty_proxy)),
            load_question_cells(),
        )
        deconfound["proxy_source"] = args.difficulty_proxy

    results = {
        "model_id": PROBE_MODEL,
        "quantization": "4-bit nf4 (12 GB VRAM; disclosed deviation from bf16)",
        "arm": "registered scale-check target (pre-reg section 3); 70B primary not run",
        "n_probe_train": int(len(y_train)),
        "n_citations": int(len(y_cit)),
        "n_fabricated": int(y_cit.sum()),
        "n_real": int((1 - y_cit).sum()),
        "layer_selected_on_probe_val": int(best_layer),
        "probe_val_auroc": val_sweep[best_layer].auroc,
        "cross_task_auroc": cross_task.auroc,
        "shuffled_label_control_auroc": shuffled.auroc,
        "shuffled_label_null_band": {
            "n_seeds": len(shuffled_aurocs),
            "min": float(min(shuffled_aurocs)),
            "mean": float(np.mean(shuffled_aurocs)),
            "max": float(max(shuffled_aurocs)),
            "aurocs": [float(a) for a in shuffled_aurocs],
            # Share of permutations at least as far from 0.5 as the observed
            # cross-task AUROC: if this is large, the cross-task figure is
            # indistinguishable from a label-blind probe on these activations.
            "share_as_extreme_as_cross_task": float(np.mean(
                [abs(a - 0.5) >= abs(cross_task.auroc - 0.5) for a in shuffled_aurocs])),
        },
        "within_domain_auroc": within.auroc,
        "h1_3_threshold": 0.70,
        "h1_3_met_by_headline": meets_h1_3(verdict.headline_auroc),
        "agreement": {
            "headline": verdict.headline,
            "headline_auroc": verdict.headline_auroc,
            "downgraded": verdict.downgraded,
            "rationale": verdict.rationale,
        },
        "difficulty_deconfound": deconfound,
        "layer_sweep_citation_auroc": {str(l): r.auroc for l, r in citation_sweep.items()},
        "layer_sweep_probe_val_auroc": {str(l): r.auroc for l, r in val_sweep.items()},
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print(f"cross-task AUROC      : {cross_task.auroc:.4f}")
    print(f"shuffled-label control: {shuffled.auroc:.4f}  (must be near 0.50)")
    print(f"within-domain AUROC   : {within.auroc:.4f}")
    print(f"agreement             : {verdict.rationale}")
    print(f"H1.3 (>= 0.70) met    : {results['h1_3_met_by_headline']}")
    if deconfound:
        sub = deconfound["high_difficulty_subset"]
        print(f"difficulty deconfound : fabrication {deconfound['fabrication_auroc']:.4f} vs "
              f"difficulty probe {deconfound['difficulty_probe_auroc']:.4f} "
              f"(proxy alone {deconfound['proxy_alone_auroc']:.4f}) -> "
              f"{deconfound['verdict']['reported_as']}")
        print(f"  high-difficulty     : fabrication {sub['fabrication_auroc']:.4f} vs "
              f"difficulty probe {sub['difficulty_probe_auroc']:.4f} on n={sub['n_test']} -> "
              f"{sub['verdict']['reported_as']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
