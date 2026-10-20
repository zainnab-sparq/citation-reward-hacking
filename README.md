# Are Fake Citations a Form of Reward Hacking?

Code and data for *Are Fake Citations a Form of Reward Hacking? A Pre-Registered Study of Post-Training and Citation Cues* (Zain Naboulsi, 2026). The manuscript source is in [`paper/`](paper/), and the pre-registration is on OSF, DOI [10.17605/OSF.IO/S53HY](https://doi.org/10.17605/OSF.IO/S53HY), with every later amendment logged in [`PRE_REGISTRATION_OSF.md`](PRE_REGISTRATION_OSF.md) §11.

## What the study does

The same 900 citation-seeking questions (academic, medical, legal) were put to 15 open-weights checkpoints from 4 model families, at each stage of post-training where the stages are released (base, SFT, DPO, final), under four prompt conditions. Every citation in every answer was extracted, checked against bibliographic and legal databases, and judged REAL, FABRICATED or UNVERIFIABLE.

- **E1** asks whether fabrication rises with post-training stage, and whether it rises most on obscure but real works.
- **E2** asks whether a prompt that expects citations raises fabrication more in post-trained checkpoints than in base ones.
- **E3** probes a model's hidden states for a fabricated-versus-real signal.
- **E4** tests whether a naive judge that scores citation surface features prefers the higher-fabrication outputs.

## Main results

Numbers are from the 15-checkpoint refit, corrected pipeline; the registered-pipeline values and the original 12-checkpoint fit sit beside them in [`paper/sections/04_results_e1_e2.tex`](paper/sections/04_results_e1_e2.tex).

- **E2 is supported.** The citation cue raises fabrication in post-trained checkpoints far more than in base ones: interaction odds ratio 6.75 [4.91, 9.29], Holm-adjusted p = 3.2e-31 ([results JSON](data/analysis/phase_c15/e2_corrected_nobootstrap.json)).
- **E1's registered interaction is null.** Fabrication rises steeply with post-training stage, but not more on obscure-but-real works: odds ratio 0.93 [0.86, 1.01], Holm-adjusted p = 0.192 ([results JSON](data/analysis/phase_c15/e1_corrected_nobootstrap.json)).
- E3 and E4 are reported in [`paper/sections/05_results_e3_e4.tex`](paper/sections/05_results_e3_e4.tex).

## The question set is not fully public

A published benchmark ends up in later training data, and then it no longer measures what it was built to measure. So, as the pre-registration commits, only the two 50-question release subsets are public:

- [`data/questions/release_50q_public_amended.jsonl`](data/questions/release_50q_public_amended.jsonl): question text only, without cell labels or expected answers.
- [`data/questions/release_50q_researcher_amended.jsonl`](data/questions/release_50q_researcher_amended.jsonl): the same questions with cell labels, expected citations and construction metadata.

The SHA-256 hashes of the full frozen set and of both subsets are in [`data/questions/QUESTIONS_HASH.txt`](data/questions/QUESTIONS_HASH.txt) and [`data/questions/DATASET_SUMMARY.md`](data/questions/DATASET_SUMMARY.md). The full set, the construction pools and the complete development history are available to reviewers on request.

Generations, verifications, judgments and the analysis frames refer to questions by ID only. Base checkpoints often repeat the prompt, and post-trained ones sometimes restate an unanswerable question's premise, so in the public generation files every run of 40 or more characters that matches a held-back question is replaced by `[QUESTION REDACTED]` padded with dots to the same length. The E1 and E2 scripts use response text only through its character count, so they reproduce exactly from the masked files. E3 and E4 read the text itself and reproduce exactly only from the full data.

## Reproducing the E1 and E2 fits

The analysis frames are in `data/analysis_inputs/<dataset>_{final,registered}/`, where `final` is the corrected pipeline and `registered` is the registered one. `phase_c` is the 12-checkpoint roster and `phase_c15` the 15-checkpoint roster. Each `generations.jsonl` ships gzipped to stay under GitHub's file-size limit; unpack them first:

```
python scripts/unpack_analysis_inputs.py
```

Then restore the R environment (R 4.6.0, packages pinned in `renv.lock`) and fit, for example E1 on 15 checkpoints, corrected pipeline, without the bootstrap:

```
Rscript -e "renv::restore()"
CITATION_BOOTSTRAP_NSIM=0 Rscript src/analysis/e1_checkpoint_cascade.R \
  --generations data/analysis_inputs/phase_c15_final/generations.jsonl \
  --verifications data/analysis_inputs/phase_c15_final/verifications.jsonl \
  --questions data/analysis_inputs/phase_c15_final/questions.jsonl \
  --models data/analysis_inputs/phase_c15_final/models.jsonl \
  --output e1_check.json
```

The output should match `data/analysis/phase_c15/e1_corrected_nobootstrap.json`. E2 is the same command with `src/analysis/e2_prompt_manipulation.R`. On Windows, `scripts/run_confirmatory.ps1 -Pass A -Dataset phase_c15` launches all four jobs; it refuses to overwrite an existing result. The registered parametric bootstrap was not run, for the measured reasons in amendment #14.

## Tests

```
uv sync --extra dev
uv run pytest
```

Seven tests need live API access and skip without it. The R suites are in `src/analysis/test_*.R`.

## Layout

| Path | Contents |
|---|---|
| `src/` | generation, citation extraction, verification and judge pipeline (Python) |
| `src/analysis/` | registered E1/E2 scripts, drift and robustness analyses, figures (R) |
| `src/e3/` | the hidden-state probe |
| `data/models/` | checkpoint roster, pinned to commit SHAs |
| `data/generations/` | raw model outputs, one file per checkpoint |
| `data/judgments/` | judge verdicts with reasoning and evidence URLs |
| `data/analysis/` | every result JSON the paper reports |
| `paper/` | LaTeX source, figures and tables |
| `docs/` | design notes, amendment drafts and decision records |

## License

Code: MIT ([`LICENSE`](LICENSE)). Data: CC BY 4.0 ([`data/LICENSE.md`](data/LICENSE.md)). Many generations contain fabricated citations by design; do not use them as references.
