"""Score the EXISTENT_WRONG_ID sub-study (PRE_REGISTRATION_OSF.md amendment #6 #3).

Reads the sub-study manifest (the frozen stratified draw + population sizes) and
the judge verdicts on that subsample, then computes the per-citation
disagreement rate vs J2a-as-reported.

Registered metric: under the amendment-#5 strict-hybrid combine, a verifier-REAL
citation re-judged stays REAL on judge-REAL and on judge-UNVERIFIABLE (verifier
fallback) and flips only on judge-FABRICATED. So "disagreement" == the judge
returns FABRICATED. Threshold 8% (Step 0 was 1/21 ≈ 4.8%, doubled).

Two estimates are reported:
  raw_fab_rate         pooled over the n=200 draw + a Wilson 95% CI. The draw
                       uses a floored (non-proportional) allocation, so this is
                       conservative (over-weights the higher-risk thin strata) —
                       the registered "upper-bound bias" framing.
  reweighted_fab_rate  sum_h (N_h / N) * r_h, the population-representative bias
                       estimate the methods supplement reports as "X%".
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

DISAGREEMENT_STATUS = "FABRICATED"
_Z_95 = NormalDist().inv_cdf(0.975)  # ~1.959963985


def wilson_ci(k: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion k/n. Returns (lo, hi).

    Valid for a simple random sample; reported here as a secondary (SRS-naive)
    summary of the pooled count. The design-correct interval for the stratified,
    non-proportional draw is the reweighted estimate ± stratified_se (below)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def stratified_se(per_stratum: list[dict[str, Any]], *, pop_total: int) -> float:
    """Design-based SE of the population-reweighted rate under stratified SRS.

    Var = Σ_h w_h² · r_h(1−r_h)/n_h · (1 − n_h/N_h), with w_h = N_h/N and the
    finite-population correction (1 − n_h/N_h) — material here because the floor
    draws 5 of 15 in the smallest strata. Cochran (1977) §5.6. This is the
    correct uncertainty for the non-proportional floored draw; the pooled Wilson
    on the same data treats it as an SRS and is anti-conservative.
    """
    if pop_total <= 0:
        return 0.0
    var = 0.0
    for s in per_stratum:
        n_h = s["n_judged"]
        if n_h <= 0:
            continue
        N_h = s["population"]
        r_h = s["n_fabricated"] / n_h
        w_h = N_h / pop_total
        fpc = 1 - (n_h / N_h) if N_h else 1.0
        var += w_h * w_h * r_h * (1 - r_h) / n_h * fpc
    return var ** 0.5


@dataclass(frozen=True)
class SubstudyResult:
    n: int
    n_judged: int
    n_missing: int
    raw_fab_rate: float
    raw_fab_ci: tuple[float, float]
    reweighted_fab_rate: float
    reweighted_ci: tuple[float, float]
    raw_unv_rate: float
    per_stratum: list[dict[str, Any]]
    gate_threshold: float
    exceeds_threshold: bool


def score_substudy(
    manifest: dict[str, Any],
    judgments: list[dict[str, Any]],
    qmap: dict[str, dict[str, Any]],
    *,
    threshold: float = 0.08,
) -> SubstudyResult:
    """Compute raw + population-reweighted disagreement rates from a sub-study
    manifest and the judge verdicts on its subsample."""
    verdict_by_cid = {
        j["custom_id"]: j.get("judge_status")
        for j in judgments
        if isinstance(j.get("custom_id"), str)
    }
    population = {(s["domain"], s["cell"]): s["population"] for s in manifest["strata"]}
    pop_total = sum(population.values())

    # Per-stratum tallies over the selected sample.
    tally: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "fab": 0, "unv": 0, "real": 0, "missing": 0}
    )
    for cid in manifest["selected_custom_ids"]:
        qid = cid.split("__", 1)[0]
        row = qmap[qid]
        stratum = (row["domain"], row["cell"])
        t = tally[stratum]
        t["n"] += 1
        status = verdict_by_cid.get(cid)
        if status == "FABRICATED":
            t["fab"] += 1
        elif status == "UNVERIFIABLE":
            t["unv"] += 1
        elif status == "REAL":
            t["real"] += 1
        else:
            t["missing"] += 1

    n = sum(t["n"] for t in tally.values())
    n_fab = sum(t["fab"] for t in tally.values())
    n_unv = sum(t["unv"] for t in tally.values())
    n_missing = sum(t["missing"] for t in tally.values())
    n_judged = n - n_missing

    # Rates exclude unreturned judgments (missing-at-random); n_missing reported.
    raw_fab_rate = (n_fab / n_judged) if n_judged else 0.0
    raw_unv_rate = (n_unv / n_judged) if n_judged else 0.0
    raw_fab_ci = wilson_ci(n_fab, n_judged)

    per_stratum = []
    for stratum in sorted(tally):
        t = tally[stratum]
        judged = t["n"] - t["missing"]
        per_stratum.append({
            "domain": stratum[0],
            "cell": stratum[1],
            "population": population.get(stratum, 0),
            "n_sample": t["n"],
            "n_judged": judged,
            "n_fabricated": t["fab"],
            "n_unverifiable": t["unv"],
            "n_real": t["real"],
            "n_missing": t["missing"],
            "fab_rate": (t["fab"] / judged) if judged else 0.0,
        })

    # Population-reweighted (post-stratified) estimate + design-based CI with FPC.
    reweighted = sum(
        (population.get((s["domain"], s["cell"]), 0) / pop_total if pop_total else 0.0)
        * s["fab_rate"]
        for s in per_stratum
    )
    se = stratified_se(per_stratum, pop_total=pop_total)
    reweighted_ci = (max(0.0, reweighted - _Z_95 * se), min(1.0, reweighted + _Z_95 * se))

    exceeds = raw_fab_rate > threshold or reweighted > threshold
    return SubstudyResult(
        n=n,
        n_judged=n_judged,
        n_missing=n_missing,
        raw_fab_rate=raw_fab_rate,
        raw_fab_ci=raw_fab_ci,
        reweighted_fab_rate=reweighted,
        reweighted_ci=reweighted_ci,
        raw_unv_rate=raw_unv_rate,
        per_stratum=per_stratum,
        gate_threshold=threshold,
        exceeds_threshold=exceeds,
    )
