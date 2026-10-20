# Dataset Summary: 900-question pre-registered set

**Frozen:** 2026-05-22 (raised from the original 600-question freeze of 2026-05-03 to clear the §3 power gate; see pre-reg §3)
**Total records:** 900 (0 with null question)
**First dataset commit:** `c85606d`

## Hashes (canonical-formatted JSONL, LF-normalized)

| File | SHA-256 |
|---|---|
| `frozen_900q.jsonl` (original, immutable) | `6c8dd0d3b6424fadf86c87295345495ef58abeeb6db0a447362bbe042fe2c614` |
| `release_50q_researcher.jsonl` (original) | `96fa2c137159b62a28d8e766b9df0daff91597dea41bc91484eab4d2cc4f9ff0` |
| `release_50q_public.jsonl` (original) | `d19770149a46832ab510ca483d9f1768a890ed395752652c933095e672333631` |
| **`frozen_900q_legal_amended.jsonl` (operative)** | `b656f033c11675abfdb23ddc965ee83e2bf01851ee297f325e61e606d017a785` |
| **`release_50q_researcher_amended.jsonl`** | `adf8cb029b6231075168d373aed333b5548cfb97b9fe0a81961846ff33dc0bf8` |
| **`release_50q_public_amended.jsonl`** | `e6cb1b46b95156c818ca0acaa0979ae2392c47af77e01c7d14bf6010b854d433` |

Hashes are mirrored in `QUESTIONS_HASH.txt` and in pre-reg header. The superseded `frozen_600q.jsonl` is left on disk for provenance.

**Amendment 2026-05-25 (round-3 panel).** The legal known-real cell originally contained 45 opinions dated ≥ 2023 (24×2023, 16×2024, 5×2025) because its recruiter filtered only on `citeCount >= 50` and omitted the pre-registered pre-2023 year gate. Those 45 were replaced with same-tier (34 Circuit, 11 SCOTUS) pre-2023 (2018–2022) opinions from the same candidate pool, by seeded (42) draw, via `src/dataset_construction/amend_legal_cutoff.py`. Slot IDs preserved; originals retained in `metadata.original_expected_citation`. The original `frozen_900q.jsonl` and its hash are retained immutably; `frozen_900q_legal_amended.jsonl` is operative for all reported analyses. See pre-reg §11 deviation log.

## Distribution (100 per domain × cell, exactly balanced)

| domain   | cell           | n   | source(s) |
|----------|---------------|-----|-----------|
| academic | known_real    | 100 | OpenAlex CS, 50 canonical (`>9999` cites) + 50 moderate (`[1000,5000]`), pre-2023 |
| academic | obscure_real  | 100 | OpenAlex CS, `[2,49]` cites, pre-2023 |
| academic | unanswerable  | 100 | Real research areas + synthetic-but-plausible authors (HalluLens MixedEntities pattern); validated via Levenshtein-3 + SPECTER v1 cosine ≥ 0.85 |
| medical  | known_real    | 100 | OpenAlex Cochrane (60%) + non-Cochrane high-cite reviews (40%), pre-2023 |
| medical  | obscure_real  | 100 | OpenAlex Medicine, `[2,49]` cites, retraction-watch-filtered, pre-2023 |
| medical  | unanswerable  | 100 | Real condition + real intervention + fictitious specific trial; validated as above |
| legal    | known_real    | 100 | CourtListener federal: SCOTUS (50%) + Circuit Courts of Appeals (50%); **pre-2023 enforced via 2026-05-25 amendment** (45 ≥2023 opinions replaced, see above) |
| legal    | obscure_real  | 100 | CourtListener federal District Courts, `citeCount [2,49]` |
| legal    | unanswerable  | 100 | Federal jurisdiction + facially-impossible vol/page numbers; Van Staphorst-trap-checked |

## Construction pipeline

1. **Recruit** raw candidates per cell (Python, deterministic with seed=42)
   - `src/dataset_construction/recruit_cli.py` (academic + medical via OpenAlex)
   - `src/dataset_construction/recruit_legal_cli.py` (legal via CourtListener v4)
   - Targets: 250+ candidates per cell to absorb pipeline attrition
2. **Filter** medical obscure-real against Retraction Watch DB (`data/external/retraction_watch_2026-05-03.csv`, 64MB)
3. **Construct** unanswerable cells via Sonnet subagents (250+ candidates per domain, working from plan §4 construction rules)
4. **Generate** natural-language questions via Sonnet subagents (one question per candidate)
5. **Validate** unanswerable candidates via two-step negative-existence proof (`src/dataset_construction/validate_unanswerable.py`):
   - Levenshtein-3 fuzzy title match against top-5 OpenAlex hits → reject
   - SPECTER v1 cosine ≥ 0.85 → reject
6. **Sample** to 900 (100 per domain × cell) with `random.Random(42)`. The 300-question top-up over the original 600 drew from the same pre-recruited pools under the same seed and the same eligibility/negative-existence filters; no criterion was relaxed. Top-up validation logged separately in `unanswerable_validation_topup.jsonl`.
7. **Hash** SHA-256 of canonical-sorted JSONL

## Validation results (unanswerable cells, deduplicated across base + top-up logs)

| domain | candidates validated | kept (passes proof) | rejected (Levenshtein) | rejected (cosine ≥ 0.85) | no OpenAlex hits (default-pass) |
|---|---|---|---|---|---|
| academic | 250 | 121 | 2 | 127 | 0 |
| medical  | 327 | 110 | 0 | 217 | 40 |
| legal    | 250 | 244 | 0 | 6 | 29 |

Each domain's kept pool (121 / 110 / 244) exceeds the 100 needed, so the unanswerable cell was sampled down to exactly 100 per domain.

The high cosine-rejection rate for academic + medical reflects the conservative-over-rejection bias (cosine ≥ 0.85 catches "same research area as a known paper," not just "same paper"). Per pre-reg §4 this is the correct direction: over-reject novel candidates that resemble existing work; never under-reject and accidentally land on a real paper.

The legal cell's low rejection rate is a known coverage limitation: OpenAlex does not index US case law, so most legal-title searches return either no hits (default-pass) or unrelated results. Mitigations:
- Construction rule enforces facially-impossible volume/page numbers (e.g., F.4th vol 999 page 8847)
- Federal-jurisdiction-only restriction (CourtListener has near-complete federal coverage)
- Van Staphorst trap checked at construction time
- **Hard Phase-B gate (pre-reg §8):** all 100 legal unanswerables are cross-checked against the CourtListener Citation Lookup API (and manually against Westlaw/Lexis) and must each return no real case before any legal fabrication rate is reported.

## Pre-reg cross-references

- §4 Materials: construction protocol matches this summary.
- §6 Statistical analysis: sensitivities (Cochrane-status, citation-tier, name-frequency) all enabled by metadata captured during recruitment.
- §8 Exclusion rules + legal-unanswerable Phase-B gate: applied at recruitment / validation / pre-report time as documented above.
- §9 Reproducibility: 50-q public version with cell labels stripped is at `release_50q_public.jsonl`; researcher version at `release_50q_researcher.jsonl`.

## Known gaps and follow-ups

1. **Legal negative-existence proof** uses OpenAlex (which does not index case law). For paper-reported numbers this is now a **hard Phase-B gate**: CourtListener Citation Lookup API + manual Westlaw/Lexis checks on all 100 legal unanswerable cases before any legal fabrication rate is reported (pre-reg §8).
2. **SPECTER calibration pair list** is currently 30 known-duplicate pairs (30/30 correctly rejected at 0.85; lowest cosine on a known-dup pair 0.912). Pre-reg target is ≥ 50; expansion to 50+ pairs is a tracked pre-OSF-submission compute task (`specter_calibration.json`).
3. **Year cutoff `pre-2023`** captures most contamination risk, but model training cutoffs vary. Sensitivity analysis stratified by year (per pre-reg §6) addresses this at analysis time. **Legal known-real exception (corrected 2026-05-25):** this cell's recruiter omitted the year gate at first freeze, admitting 45 ≥2023 opinions; corrected by the amendment above. The legal year-stratified sensitivity is elevated to co-primary for this cell, reported on both original and amended sets.
4. **Null-question records** during initial Q-gen (titles too garbled for question synthesis) were filtered out before sampling; the frozen set has 0 null-question records.

## Reproducibility

Re-running the pipeline from the same `OPENALEX_MAILTO` with the same seeds reproduces the recruitment portion deterministically. The unanswerable construction step is non-deterministic (LLM-driven) but the validation step is deterministic given the input file. The post-validation sampling uses `random.Random(42)` so the final 900-question selection is deterministic given the validated pool.

For exact reproduction, freeze the inputs at:
- `data/questions/raw/*.jsonl` (recruited + constructed candidates)
- `data/questions/with_q/*.jsonl` (after question generation)
- `data/external/retraction_watch_2026-05-03.csv` (snapshot of the Retraction Watch DB)
- `data/questions/specter_calibration.json` (calibration result)
- Plan-file git SHA at A.6 freeze
