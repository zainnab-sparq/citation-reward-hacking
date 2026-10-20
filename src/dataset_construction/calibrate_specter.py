"""SPECTER cosine threshold calibration (pre-reg §4 commitment).

Verifies that the chosen threshold (0.85) rejects 0/N known-duplicate title
pairs as "unanswerable" (i.e., correctly catches all of them at step 2 of the
negative-existence proof).

Pair list: curated arXiv-vs-published title pairs for well-known ML / scientific
papers. Pairs are deliberately a mix of identical-title (sanity check) and
slightly-varied-title (the real test of cosine robustness).

Run:
    .venv/Scripts/python.exe -m src.dataset_construction.calibrate_specter

Outputs `data/questions/specter_calibration.json` for the pre-reg supplement.
Add more pairs before A.6 freeze to reach the >= 50 target.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from src.dataset_construction.specter_embedder import SpecterEmbedder

THRESHOLD = 0.85


@dataclass
class TitlePair:
    label: str
    title_a: str  # e.g., arXiv version
    title_b: str  # e.g., published-venue version


# Each pair is the SAME paper, two title surface forms. SPECTER must score
# these as cosine >= 0.85 — otherwise the 0.85 threshold would incorrectly
# pass the "unanswerable" check for what is in fact a real existing work.
KNOWN_DUPLICATES: list[TitlePair] = [
    TitlePair("BERT", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"),
    TitlePair("BERT-shortened", "BERT: Pre-training of Deep Bidirectional Transformers", "BERT Pre-training of Deep Bidirectional Transformers for Language Understanding"),
    TitlePair("Attention", "Attention Is All You Need", "Attention is all you need"),
    TitlePair("GPT-3", "Language Models are Few-Shot Learners", "Language Models Are Few-Shot Learners"),
    TitlePair("ResNet", "Deep Residual Learning for Image Recognition", "Deep Residual Learning for Image Recognition"),
    TitlePair("ResNet-varied", "Deep Residual Learning for Image Classification", "Deep Residual Learning for Image Recognition"),
    TitlePair("Adam", "Adam: A Method for Stochastic Optimization", "Adam: A Method for Stochastic Optimization"),
    TitlePair("Adam-varied", "Adam Optimizer for Stochastic Gradient Descent", "Adam: A Method for Stochastic Optimization"),
    TitlePair("Dropout", "Dropout: A Simple Way to Prevent Neural Networks from Overfitting", "Dropout: a simple way to prevent neural networks from overfitting"),
    TitlePair("BatchNorm", "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift", "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"),
    TitlePair("BatchNorm-varied", "Batch Normalization for Faster Deep Network Training", "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift"),
    TitlePair("VGG", "Very Deep Convolutional Networks for Large-Scale Image Recognition", "Very Deep Convolutional Networks for Large-Scale Image Recognition"),
    TitlePair("GAN", "Generative Adversarial Networks", "Generative Adversarial Nets"),
    TitlePair("VAE", "Auto-Encoding Variational Bayes", "Auto-Encoding Variational Bayes"),
    TitlePair("Seq2Seq", "Sequence to Sequence Learning with Neural Networks", "Sequence to Sequence Learning with Neural Networks"),
    TitlePair("word2vec", "Efficient Estimation of Word Representations in Vector Space", "Efficient Estimation of Word Representations in Vector Space"),
    TitlePair("AlphaFold", "Highly accurate protein structure prediction with AlphaFold", "Highly accurate protein structure prediction with AlphaFold"),
    TitlePair("CLIP", "Learning Transferable Visual Models From Natural Language Supervision", "Learning Transferable Visual Models from Natural Language Supervision"),
    TitlePair("LoRA", "LoRA: Low-Rank Adaptation of Large Language Models", "LoRA: Low-Rank Adaptation of Large Language Models"),
    TitlePair("LoRA-varied", "Low-Rank Adaptation of Large Language Models", "LoRA: Low-Rank Adaptation of Large Language Models"),
    TitlePair("Chinchilla", "Training Compute-Optimal Large Language Models", "Training Compute-Optimal Large Language Models"),
    TitlePair("RLHF-Christiano", "Deep Reinforcement Learning from Human Preferences", "Deep reinforcement learning from human preferences"),
    TitlePair("InstructGPT", "Training language models to follow instructions with human feedback", "Training language models to follow instructions with human feedback"),
    TitlePair("CoT", "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models", "Chain of Thought Prompting Elicits Reasoning in Large Language Models"),
    TitlePair("ImageNet", "ImageNet Classification with Deep Convolutional Neural Networks", "ImageNet classification with deep convolutional neural networks"),
    TitlePair("LSTM", "Long Short-Term Memory", "Long Short-term Memory"),
    TitlePair("U-Net", "U-Net: Convolutional Networks for Biomedical Image Segmentation", "U-Net: Convolutional Networks for Biomedical Image Segmentation"),
    TitlePair("PPO", "Proximal Policy Optimization Algorithms", "Proximal Policy Optimization Algorithms"),
    TitlePair("TRPO", "Trust Region Policy Optimization", "Trust Region Policy Optimization"),
    TitlePair("Sycophancy", "Towards Understanding Sycophancy in Language Models", "Towards Understanding Sycophancy in Language Models"),
]

# Pairs of DIFFERENT papers that are nominally similar — for the false-positive
# leg. SPECTER should score these as cosine < 0.85 (or we have a problem).
DIFFERENT_PAPERS: list[TitlePair] = [
    TitlePair("BERT vs RoBERTa",
              "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
              "RoBERTa: A Robustly Optimized BERT Pretraining Approach"),
    TitlePair("ResNet vs DenseNet",
              "Deep Residual Learning for Image Recognition",
              "Densely Connected Convolutional Networks"),
    TitlePair("Adam vs SGD",
              "Adam: A Method for Stochastic Optimization",
              "Stochastic Gradient Descent with Momentum"),
    TitlePair("VAE vs GAN",
              "Auto-Encoding Variational Bayes",
              "Generative Adversarial Networks"),
    TitlePair("LoRA vs QLoRA",
              "LoRA: Low-Rank Adaptation of Large Language Models",
              "QLoRA: Efficient Finetuning of Quantized LLMs"),
    TitlePair("GPT-3 vs Chinchilla",
              "Language Models are Few-Shot Learners",
              "Training Compute-Optimal Large Language Models"),
]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def main() -> None:
    embedder = SpecterEmbedder()
    print(f"Loading SPECTER model {embedder.__class__.__name__}...")
    # Force the load now so the first batch isn't ambiguously timed.
    _ = embedder.model

    print(f"\nKnown-duplicate pairs (target cosine >= {THRESHOLD}):")
    dup_results = []
    for pair in KNOWN_DUPLICATES:
        emb = embedder.embed([pair.title_a, pair.title_b])
        cos = _cosine(emb[0], emb[1])
        flag = "OK" if cos >= THRESHOLD else "BELOW THRESHOLD"
        dup_results.append({"label": pair.label, "cosine": cos, "passes": cos >= THRESHOLD,
                            "title_a": pair.title_a, "title_b": pair.title_b})
        print(f"  {pair.label:<20} cos={cos:.3f}  {flag}")

    print(f"\nDifferent-paper pairs (target cosine < {THRESHOLD}):")
    diff_results = []
    for pair in DIFFERENT_PAPERS:
        emb = embedder.embed([pair.title_a, pair.title_b])
        cos = _cosine(emb[0], emb[1])
        flag = "OK" if cos < THRESHOLD else "FALSE POSITIVE (would be rejected)"
        diff_results.append({"label": pair.label, "cosine": cos, "below_threshold": cos < THRESHOLD,
                             "title_a": pair.title_a, "title_b": pair.title_b})
        print(f"  {pair.label:<20} cos={cos:.3f}  {flag}")

    n_dup = len(dup_results)
    n_dup_pass = sum(1 for r in dup_results if r["passes"])
    n_diff = len(diff_results)
    n_diff_below = sum(1 for r in diff_results if r["below_threshold"])

    print(f"\nSummary:")
    print(f"  Known-duplicates passing >= {THRESHOLD}: {n_dup_pass}/{n_dup}")
    print(f"  Different-papers below   < {THRESHOLD}: {n_diff_below}/{n_diff}")
    if n_dup_pass < n_dup:
        min_cos = min(r["cosine"] for r in dup_results)
        print(f"  WARNING: lowest known-duplicate cosine = {min_cos:.3f}; consider lowering threshold or removing the failing pair.")

    out_path = Path("data/questions/specter_calibration.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "model": "sentence-transformers/allenai-specter",
        "threshold": THRESHOLD,
        "n_known_duplicates": n_dup,
        "n_known_duplicates_passing": n_dup_pass,
        "n_different_papers": n_diff,
        "n_different_papers_below_threshold": n_diff_below,
        "known_duplicates": dup_results,
        "different_papers": diff_results,
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
