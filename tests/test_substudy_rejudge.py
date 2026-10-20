"""Contract tests for src/analysis/substudy_rejudge.py.

The paired re-judge sub-study samples already-judged citations and judges them
AGAIN under the current instrument, keeping both verdicts. Because the original
prompt files are all still on disk, the re-judge reads the BYTE-IDENTICAL prompt,
so any verdict difference is attributable to the tool environment alone.

The properties that make it a valid paired design, and that this module pins:

  * it NEVER writes to the main judgments checkpoint (which is append-only and
    has no dedup-on-write, so a stray write would corrupt the main dataset);
  * it samples from BOTH the EARLY and DEGRADED eras, since re-judging only the
    degraded middle would miss that EARLY is 52% of the corpus;
  * sampling is stratified by condition, because differentiality BY CONDITION is
    the whole question the sub-study exists to answer;
  * it is deterministic given a seed, and resumable (already-re-judged ids are
    excluded), so an interrupted night does not corrupt the sample;
  * the emitted batch script is LF-only. CRLF in a batch script is rejected by
    the Workflow permission handler, a bug that has cost this project a night
    before.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.substudy_rejudge import (
    SUBSTUDY_SEED,
    allocate_by_condition,
    allocate_equally,
    build_batch_script,
    prompt_path_for,
    sample_rejudge_items,
)
from src.analysis.judge_era import load_judgment_meta


CONDITIONS = ("neutral", "citations_expected", "abstention_permitted", "length_matched")


def _judgment(row: int, condition: str) -> dict:
    return {
        "custom_id": (
            f"academic_obscure_real_{row:04d}__allenai/Llama-3.1-Tulu-3-8B"
            f"__{condition}__t0__c0"
        ),
        "judge_status": "FABRICATED",
        "judge_reasoning": "r",
        "judge_evidence_url": "",
        "judge_search_count": 2,
        "error": None,
    }


def _corpus(tmp_path: Path, n: int = 26_000) -> Path:
    """A judgments file long enough to span the EARLY and DEGRADED eras."""
    path = tmp_path / "judgments.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for i in range(n):
            fh.write(json.dumps(_judgment(i, CONDITIONS[i % 4])) + "\n")
    return path


@pytest.fixture
def meta(tmp_path: Path) -> list[dict]:
    return load_judgment_meta(_corpus(tmp_path))


# ---------- allocation ----------


def test_allocate_by_condition_sums_to_the_requested_n() -> None:
    counts = {"neutral": 100, "citations_expected": 300, "length_matched": 100}
    alloc = allocate_by_condition(counts, 50)
    assert sum(alloc.values()) == 50


def test_allocate_by_condition_is_proportional() -> None:
    counts = {"a": 800, "b": 200}
    alloc = allocate_by_condition(counts, 100)
    assert alloc["a"] == 80
    assert alloc["b"] == 20


def test_allocate_by_condition_never_exceeds_available() -> None:
    counts = {"a": 3, "b": 1000}
    alloc = allocate_by_condition(counts, 100)
    assert alloc["a"] <= 3


def test_allocate_equally_balances_the_strata() -> None:
    """Balanced allocation is the powerful design for a per-condition contrast:
    it maximises the SMALLEST cell, and the smallest cell is what limits the
    comparison. Proportional allocation would leave abstention_permitted at ~29."""
    alloc = allocate_equally({"a": 500, "b": 500, "c": 500, "d": 500}, 250)
    assert set(alloc.values()) == {62, 63}
    assert sum(alloc.values()) == 250


def test_allocate_equally_redistributes_from_exhausted_strata() -> None:
    alloc = allocate_equally({"a": 5, "b": 1000, "c": 1000, "d": 1000}, 200)
    assert alloc["a"] == 5
    assert sum(alloc.values()) == 200


# ---------- design weights ----------


def test_sampled_items_carry_a_design_weight(meta: list[dict]) -> None:
    """Balanced sampling over unbalanced strata needs weights to recover any
    corpus-level rate. Same machinery as the Phase B EXISTENT_WRONG_ID sub-study."""
    items = sample_rejudge_items(meta, n_per_era=100)
    assert all(item["design_weight"] > 0 for item in items)


def test_design_weight_is_stratum_population_over_sample(meta: list[dict]) -> None:
    items = sample_rejudge_items(meta, n_per_era=100)
    early_neutral = [
        i for i in items if i["era"] == "EARLY" and i["condition"] == "neutral"
    ]
    population = sum(
        1 for m in meta if m["era"] == "EARLY" and m["condition"] == "neutral"
    )
    assert early_neutral[0]["design_weight"] == pytest.approx(
        population / len(early_neutral)
    )


# ---------- sampling ----------


def test_sample_draws_from_both_early_and_degraded(meta: list[dict]) -> None:
    """EARLY is 52% of the corpus, so a DEGRADED-only sub-study would miss the
    era that matters most."""
    items = sample_rejudge_items(meta, n_per_era=100)
    eras = {item["era"] for item in items}
    assert eras == {"EARLY", "DEGRADED"}


def test_sample_never_draws_from_the_recovered_era(meta: list[dict]) -> None:
    """RECOVERED rows were produced by the CURRENT instrument; re-judging them
    measures noise, not drift."""
    items = sample_rejudge_items(meta, n_per_era=100)
    assert all(item["era"] != "RECOVERED" for item in items)


def test_sample_respects_n_per_era(meta: list[dict]) -> None:
    items = sample_rejudge_items(meta, n_per_era=100)
    assert sum(1 for i in items if i["era"] == "EARLY") == 100
    assert sum(1 for i in items if i["era"] == "DEGRADED") == 100


def test_sample_is_stratified_across_all_conditions(meta: list[dict]) -> None:
    items = sample_rejudge_items(meta, n_per_era=100)
    early = [i for i in items if i["era"] == "EARLY"]
    assert {i["condition"] for i in early} == set(CONDITIONS)


def test_sample_is_deterministic_given_the_seed(meta: list[dict]) -> None:
    a = sample_rejudge_items(meta, n_per_era=50, seed=SUBSTUDY_SEED)
    b = sample_rejudge_items(meta, n_per_era=50, seed=SUBSTUDY_SEED)
    assert [i["custom_id"] for i in a] == [i["custom_id"] for i in b]


def test_a_different_seed_gives_a_different_sample(meta: list[dict]) -> None:
    a = sample_rejudge_items(meta, n_per_era=50, seed=1)
    b = sample_rejudge_items(meta, n_per_era=50, seed=2)
    assert [i["custom_id"] for i in a] != [i["custom_id"] for i in b]


def test_sample_excludes_already_rejudged_ids(meta: list[dict]) -> None:
    """Resume safety: an interrupted night must not re-draw completed items."""
    first = sample_rejudge_items(meta, n_per_era=50)
    done = {i["custom_id"] for i in first}
    second = sample_rejudge_items(meta, n_per_era=50, exclude=done)
    assert not done & {i["custom_id"] for i in second}


def test_sampled_items_carry_the_original_verdict_for_pairing(meta: list[dict]) -> None:
    """Pairing is the point: the original verdict must travel with the item."""
    item = sample_rejudge_items(meta, n_per_era=10)[0]
    assert item["original_status"] == "FABRICATED"
    assert "row_index" in item


# ---------- prompt paths ----------


def test_prompt_path_matches_the_judge_pipeline_naming(tmp_path: Path) -> None:
    """Must resolve to the SAME file the original judgment read, or the design
    stops being paired."""
    from src.judge_via_workflow import _prompt_filename

    cid = "academic_obscure_real_0100__allenai/OLMo-2-1124-13B__neutral__t0__c0"
    assert Path(prompt_path_for(cid, tmp_path)).name == _prompt_filename(cid)


# ---------- batch script ----------


def test_batch_script_is_lf_only(tmp_path: Path) -> None:
    """CRLF is rejected by the Workflow permission handler."""
    items = [{"custom_id": "a__b__neutral__t0__c0", "prompt_path": str(tmp_path / "a.txt")}]
    assert "\r" not in build_batch_script(items, "rejudge-000")


def test_batch_script_embeds_items_and_pins_sonnet(tmp_path: Path) -> None:
    items = [{"custom_id": "a__b__neutral__t0__c0", "prompt_path": str(tmp_path / "a.txt")}]
    script = build_batch_script(items, "rejudge-000")
    assert "export const meta" in script
    assert "a__b__neutral__t0__c0" in script
    assert "'sonnet'" in script


def test_batch_script_embeds_every_item(tmp_path: Path) -> None:
    items = [
        {"custom_id": f"q{i}__m__neutral__t0__c0", "prompt_path": str(tmp_path / f"{i}.txt")}
        for i in range(5)
    ]
    script = build_batch_script(items, "rejudge-000")
    assert all(f"q{i}__m__neutral__t0__c0" in script for i in range(5))
