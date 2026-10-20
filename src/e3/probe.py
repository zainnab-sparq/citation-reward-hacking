"""The E3 linear probe and the pre-registered rules that read its output.

The probe itself is deliberately dull: standardize, fit a logistic regression,
score AUROC. H1.3 is a claim about *linear decodability*, so anything more
expressive would answer a different question than the one registered.

What is not dull is the set of decision rules around it. Each one is written
here as code with a test rather than as a note in a document, because every one
of them is a rule about how to read a number you are looking at only after you
already know what the number is:

  - `apply_agreement_rule` decides whether the cross-task probe or the
    within-domain probe is the headline estimate.
  - `difficulty_deconfound_verdict` decides whether the signal may be reported
    as fabrication-decodability at all, or only as difficulty-decodability.
  - `shuffled_label_control` is the leak detector. If it does not fall to
    chance, nothing else on this page means anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Pre-registration section 2, H1.3.
H1_3_AUROC_THRESHOLD = 0.70

# Pre-registration section 6, E3 guardrails: "if the within-domain probe AUROC
# exceeds the cross-task probe AUROC by more than 10pp".
AGREEMENT_TOLERANCE = 0.10

# The registered rule is a STRICT inequality, so a margin of exactly 10pp must
# not downgrade. In binary floating point 0.80 - 0.70 is 0.10000000000000009,
# which is greater than 0.10, so a naive `>` downgrades the headline estimate at
# the exact boundary the registration says it should keep. Compare with a
# tolerance far below any AUROC difference that could be meaningful.
_BOUNDARY_EPS = 1e-9

MAX_ITER = 2000

# Ridge penalty for the difficulty probe. The hidden states have 4096 dims and
# the citation split has ~2,300 training rows, so an unpenalized regression
# would interpolate; the default sklearn strength is kept explicit here.
RIDGE_ALPHA = 1.0


@dataclass(frozen=True)
class ProbeResult:
    auroc: float
    n_train: int
    n_test: int


@dataclass(frozen=True)
class AgreementVerdict:
    headline: str  # "cross_task" or "within_domain"
    headline_auroc: float
    downgraded: bool
    rationale: str


@dataclass(frozen=True)
class DifficultyVerdict:
    reads_difficulty: bool
    reported_as: str
    margin: float


def _validate(X_train, y_train, X_test, y_test):
    X_train, y_train = np.asarray(X_train), np.asarray(y_train)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            f"train rows disagree: X has {X_train.shape[0]}, y has {y_train.shape[0]}"
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"test rows disagree: X has {X_test.shape[0]}, y has {y_test.shape[0]}"
        )
    if len(np.unique(y_train)) < 2:
        raise ValueError("training labels contain only one class; AUROC is undefined")
    if len(np.unique(y_test)) < 2:
        raise ValueError("test labels contain only one class; AUROC is undefined")
    return X_train, y_train, X_test, y_test


def fit_probe(X_train, y_train, X_test, y_test, seed: int = 0) -> ProbeResult:
    """Fit the linear probe and score AUROC on held-out activations."""
    X_train, y_train, X_test, y_test = _validate(X_train, y_train, X_test, y_test)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=MAX_ITER, random_state=seed),
    )
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    return ProbeResult(
        auroc=float(roc_auc_score(y_test, scores)),
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
    )


def shuffled_label_control(X_train, y_train, X_test, y_test, seed: int = 0) -> ProbeResult:
    """Refit on permuted training labels; must land near 0.50 or something leaks."""
    y_shuffled = np.asarray(y_train).copy()
    np.random.default_rng(seed).shuffle(y_shuffled)
    return fit_probe(X_train, y_shuffled, X_test, y_test, seed=seed)


def sweep_layers(train_layers: dict, test_layers: dict) -> dict[int, ProbeResult]:
    """Per-layer ablation: one probe per hidden layer, all layers reported.

    Reporting every layer rather than the best one is the point. A single
    best-layer AUROC selected after the fact is a maximum over 32 correlated
    tries, not an estimate.
    """
    missing = set(train_layers) ^ set(test_layers)
    if missing:
        raise ValueError(f"train and test layer sets differ: {sorted(missing)}")
    out = {}
    for layer in sorted(train_layers):
        X_tr, y_tr = train_layers[layer]
        X_te, y_te = test_layers[layer]
        out[layer] = fit_probe(X_tr, y_tr, X_te, y_te)
    return out


def apply_agreement_rule(
    cross_task_auroc: float, within_domain_auroc: float
) -> AgreementVerdict:
    """Pre-committed agreement rule (section 6, E3 cross-task-transfer assumption).

    A within-domain probe that beats the cross-task probe by more than 10pp is
    evidence the cross-task probe was partly reading task format rather than
    fabrication, so the within-domain probe takes over as the headline.
    """
    margin = within_domain_auroc - cross_task_auroc
    if margin > AGREEMENT_TOLERANCE + _BOUNDARY_EPS:
        return AgreementVerdict(
            headline="within_domain",
            headline_auroc=within_domain_auroc,
            downgraded=True,
            rationale=(
                f"within-domain exceeds cross-task by {margin:.3f} > "
                f"{AGREEMENT_TOLERANCE:.2f}; cross-task result downgraded to "
                "'may be partly reading task-format'"
            ),
        )
    return AgreementVerdict(
        headline="cross_task",
        headline_auroc=cross_task_auroc,
        downgraded=False,
        rationale=(
            f"probes agree within {AGREEMENT_TOLERANCE:.2f} "
            f"(margin {margin:+.3f}); cross-task result stands as headline"
        ),
    )


def _oriented_auroc(y_true, scores) -> float:
    """AUROC in whichever direction is larger.

    The difficulty proxy has an arbitrary sign relative to the fabrication
    label, so a probe that reads it perfectly could land at 0.10 or 0.90. Both
    are the same evidence that the hidden states carry difficulty, and taking
    the larger orientation is conservative for the fabrication claim.
    """
    auroc = float(roc_auc_score(y_true, scores))
    return max(auroc, 1.0 - auroc)


def difficulty_probe(X_train, d_train, X_test, y_test) -> ProbeResult:
    """Guardrail (a): a probe that sees only what hidden states say about difficulty.

    Ridge regression from hidden states to the continuous per-question proxy on
    the training rows; the predicted proxy on the test rows is then scored as a
    fabricated-vs-real classifier. The fabrication probe must beat this number
    on the same test rows before its signal may be called fabrication.
    """
    X_train, d_train = np.asarray(X_train), np.asarray(d_train, dtype=float)
    X_test, y_test = np.asarray(X_test), np.asarray(y_test)
    if X_train.shape[0] != d_train.shape[0]:
        raise ValueError(
            f"train rows disagree: X has {X_train.shape[0]}, proxy has {d_train.shape[0]}"
        )
    if X_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"test rows disagree: X has {X_test.shape[0]}, y has {y_test.shape[0]}"
        )
    if len(np.unique(y_test)) < 2:
        raise ValueError("test labels contain only one class; AUROC is undefined")
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(X_train, d_train)
    predicted = model.predict(X_test)
    return ProbeResult(
        auroc=_oriented_auroc(y_test, predicted),
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
    )


def proxy_alone_auroc(d_test, y_test) -> float:
    """How well the raw proxy, with no hidden states at all, separates the labels."""
    return _oriented_auroc(np.asarray(y_test), np.asarray(d_test, dtype=float))


def difficulty_deconfound_verdict(
    fabrication_auroc: float, difficulty_auroc: float
) -> DifficultyVerdict:
    """Item-level difficulty deconfound (section 6, E3 guardrail (a)).

    If the fabrication probe is no better than a probe trained on the item
    difficulty proxy, the signal is difficulty-decodability and must be reported
    as such.
    """
    margin = fabrication_auroc - difficulty_auroc
    if margin <= 0:
        return DifficultyVerdict(
            reads_difficulty=True,
            reported_as="difficulty-decodability, not fabrication-decodability",
            margin=margin,
        )
    return DifficultyVerdict(
        reads_difficulty=False,
        reported_as="fabrication-decodability (out-performs the difficulty proxy)",
        margin=margin,
    )


def meets_h1_3(auroc: float) -> bool:
    """H1.3 is declared at AUROC >= 0.70; this is that comparison, once."""
    return auroc >= H1_3_AUROC_THRESHOLD
