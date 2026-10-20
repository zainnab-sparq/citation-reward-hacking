"""Contract tests for src/extract_batch.py (Phase B citation extraction).

Mirrors elicit.py's batch layer for the Sonnet-driven structured-output
citation extractor. Input: a generations.jsonl (the output of
generate_via_batch_api) + the questions JSONL for domain lookup.
Output: extractions.jsonl with one record per generation, each holding
the structured-citation list returned by Sonnet 4.6's record_citations
tool call.

Same key shape as BatchKey (question_id, model_snapshot, condition,
trial_id) so the aggregation layer can join extractions to generations
1:1 by custom_id.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.elicit import BatchKey


# ---------- ExtractionRecord ----------


def test_extraction_record_to_jsonl_dict_roundtrips():
    from src.extract_batch import ExtractionRecord

    record = ExtractionRecord(
        question_id="academic_known_real_0000",
        condition="neutral",
        model_id="claude-haiku-4-5-20251001",
        trial_id=2,
        extractor_model_id="claude-sonnet-4-6",
        extracted_at="2026-06-04T05:00:00+00:00",
        citations=[
            {"raw_span": "Vaswani et al. (2017)...", "title": "Attention is all you need",
             "authors": ["Vaswani"], "year": 2017, "venue": "NeurIPS",
             "doi": None, "case_name": None, "reporter": None},
        ],
    )
    d = record.to_jsonl_dict()
    assert d["question_id"] == "academic_known_real_0000"
    assert d["model_id"] == "claude-haiku-4-5-20251001"
    assert d["trial_id"] == 2
    assert len(d["citations"]) == 1
    assert d["citations"][0]["title"] == "Attention is all you need"


# ---------- build_extraction_requests ----------


@pytest.fixture
def two_generations() -> list[dict]:
    return [
        {
            "question_id": "academic_known_real_0001",
            "condition": "citations_expected",
            "model_id": "claude-haiku-4-5-20251001",
            "output_text": "The Transformer was introduced by Vaswani et al. (2017).",
            "input_tokens": 30,
            "output_tokens": 50,
            "finish_reason": "end_turn",
            "timestamp": "2026-06-04T04:00:00+00:00",
            "trial_id": 0,
        },
        {
            "question_id": "medical_obscure_real_0007",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "output_text": "See Smith JK et al., Lancet 2019.",
            "input_tokens": 30,
            "output_tokens": 40,
            "finish_reason": "end_turn",
            "timestamp": "2026-06-04T04:00:01+00:00",
            "trial_id": 1,
        },
    ]


@pytest.fixture
def question_domain_map() -> dict[str, str]:
    return {
        "academic_known_real_0001": "academic",
        "medical_obscure_real_0007": "medical",
    }


def test_build_extraction_requests_one_per_generation(two_generations, question_domain_map):
    from src.extract_batch import build_extraction_requests

    reqs = build_extraction_requests(
        generations=two_generations,
        question_domain=question_domain_map,
        extractor_model_id="claude-sonnet-4-6",
        max_output_tokens=2048,
    )
    assert len(reqs) == 2


def test_build_extraction_requests_custom_id_matches_batchkey(
    two_generations, question_domain_map
):
    """custom_id must round-trip through BatchKey.decode() so aggregation can join."""
    from src.extract_batch import build_extraction_requests

    reqs = build_extraction_requests(
        generations=two_generations,
        question_domain=question_domain_map,
        extractor_model_id="claude-sonnet-4-6",
        max_output_tokens=2048,
    )
    first_key = BatchKey.decode(reqs[0]["custom_id"])
    assert first_key.question_id == "academic_known_real_0001"
    assert first_key.model_snapshot == "claude-haiku-4-5-20251001"
    assert first_key.condition == "citations_expected"
    assert first_key.trial_id == 0


def test_build_extraction_requests_params_use_extractor_model_not_generator(
    two_generations, question_domain_map
):
    """params.model must be the Sonnet extractor, NOT the Haiku generator."""
    from src.extract_batch import build_extraction_requests

    reqs = build_extraction_requests(
        generations=two_generations,
        question_domain=question_domain_map,
        extractor_model_id="claude-sonnet-4-6",
        max_output_tokens=2048,
    )
    for r in reqs:
        assert r["params"]["model"] == "claude-sonnet-4-6"
        assert r["params"]["max_tokens"] == 2048


def test_build_extraction_requests_forces_record_citations_tool_use(
    two_generations, question_domain_map
):
    from src.extract_batch import build_extraction_requests

    reqs = build_extraction_requests(
        generations=two_generations,
        question_domain=question_domain_map,
        extractor_model_id="claude-sonnet-4-6",
        max_output_tokens=2048,
    )
    for r in reqs:
        tools = r["params"]["tools"]
        assert any(t["name"] == "record_citations" for t in tools)
        assert r["params"]["tool_choice"] == {"type": "tool", "name": "record_citations"}


def test_build_extraction_requests_includes_response_text_and_domain(
    two_generations, question_domain_map
):
    from src.extract_batch import build_extraction_requests

    reqs = build_extraction_requests(
        generations=two_generations,
        question_domain=question_domain_map,
        extractor_model_id="claude-sonnet-4-6",
        max_output_tokens=2048,
    )
    medical_req = next(r for r in reqs if "Smith JK" in r["params"]["messages"][0]["content"])
    content = medical_req["params"]["messages"][0]["content"]
    assert "medical" in content
    assert "Smith JK et al., Lancet 2019." in content


def test_build_extraction_requests_raises_on_unknown_question(
    two_generations, question_domain_map
):
    from src.extract_batch import build_extraction_requests

    two_generations.append(
        {**two_generations[0], "question_id": "ghost-q-999"}
    )
    with pytest.raises(KeyError, match="ghost-q-999"):
        build_extraction_requests(
            generations=two_generations,
            question_domain=question_domain_map,
            extractor_model_id="claude-sonnet-4-6",
            max_output_tokens=2048,
        )


def test_build_extraction_requests_skips_error_generations(question_domain_map):
    """Generation rows carrying an `error` field have no output_text to extract from."""
    from src.extract_batch import build_extraction_requests

    gens = [
        {
            "question_id": "academic_known_real_0001",
            "condition": "neutral",
            "model_id": "claude-haiku-4-5-20251001",
            "output_text": "",
            "input_tokens": None,
            "output_tokens": None,
            "finish_reason": None,
            "timestamp": "2026-06-04T04:00:00+00:00",
            "trial_id": 0,
            "error": "batch_errored",
        },
    ]
    reqs = build_extraction_requests(
        generations=gens,
        question_domain=question_domain_map,
        extractor_model_id="claude-sonnet-4-6",
        max_output_tokens=2048,
    )
    assert reqs == []


# ---------- parse_extraction_results ----------


def _succeeded_tool_use(custom_id: str, citations: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="tool_use",
                        name="record_citations",
                        input={"citations": citations},
                    ),
                ],
            ),
        ),
    )


def _errored(custom_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        custom_id=custom_id,
        result=SimpleNamespace(type="errored", error=SimpleNamespace(message="rate")),
    )


def test_parse_extraction_results_yields_one_record_per_succeeded(
    two_generations, question_domain_map
):
    from src.extract_batch import build_extraction_requests, parse_extraction_results

    reqs = build_extraction_requests(
        generations=two_generations,
        question_domain=question_domain_map,
        extractor_model_id="claude-sonnet-4-6",
        max_output_tokens=2048,
    )
    items = [
        _succeeded_tool_use(reqs[0]["custom_id"], [
            {"raw_span": "Vaswani 2017", "title": "Attention", "authors": ["Vaswani"], "year": 2017},
        ]),
        _succeeded_tool_use(reqs[1]["custom_id"], [
            {"raw_span": "Smith Lancet", "title": "Lancet", "authors": ["Smith"], "year": 2019},
        ]),
    ]
    records = list(
        parse_extraction_results(
            items,
            extractor_model_id="claude-sonnet-4-6",
            extracted_at="2026-06-04T05:00:00+00:00",
        )
    )
    assert len(records) == 2
    by_q = {r.question_id: r for r in records}
    assert by_q["academic_known_real_0001"].citations[0]["title"] == "Attention"
    assert by_q["medical_obscure_real_0007"].condition == "neutral"
    assert by_q["academic_known_real_0001"].extractor_model_id == "claude-sonnet-4-6"
    assert by_q["academic_known_real_0001"].extracted_at == "2026-06-04T05:00:00+00:00"


def test_parse_extraction_results_errored_yields_record_with_empty_citations():
    """An errored extractor call still emits a row (so the aggregator sees the gen)."""
    from src.extract_batch import parse_extraction_results

    cid = BatchKey("academic_known_real_0001", "claude-haiku-4-5-20251001", "neutral", 0).encode()
    items = [_errored(cid)]
    records = list(
        parse_extraction_results(
            items,
            extractor_model_id="claude-sonnet-4-6",
            extracted_at="2026-06-04T05:00:00+00:00",
        )
    )
    assert len(records) == 1
    assert records[0].citations == []
    assert records[0].error == "rate"


def test_parse_extraction_results_text_only_response_yields_empty_citations():
    """If Sonnet replied with text instead of a tool_use block — record 0 citations."""
    from src.extract_batch import parse_extraction_results

    cid = BatchKey("academic_known_real_0001", "claude-haiku-4-5-20251001", "neutral", 0).encode()
    items = [
        SimpleNamespace(
            custom_id=cid,
            result=SimpleNamespace(
                type="succeeded",
                message=SimpleNamespace(content=[
                    SimpleNamespace(type="text", text="I refuse."),
                ]),
            ),
        )
    ]
    records = list(
        parse_extraction_results(
            items,
            extractor_model_id="claude-sonnet-4-6",
            extracted_at="2026-06-04T05:00:00+00:00",
        )
    )
    assert len(records) == 1
    assert records[0].citations == []


# ---------- Helper: load generations + question_domain from JSONL ----------


def test_load_generations_reads_jsonl(tmp_path: Path, two_generations):
    from src.extract_batch import load_generations

    path = tmp_path / "g.jsonl"
    path.write_text(
        "\n".join(json.dumps(g) for g in two_generations) + "\n",
        encoding="utf-8",
    )
    gens = load_generations(path)
    assert len(gens) == 2
    assert gens[0]["question_id"] == "academic_known_real_0001"


def test_load_question_domain_map_from_phase_b_questions(tmp_path: Path):
    """The extractor needs question_id -> domain. Builds from the frozen
    Phase B JSONL via load_phase_b_questions (which reads `question` field)."""
    from src.extract_batch import build_question_domain_map_from_path

    path = tmp_path / "q.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "academic_known_real_0000",
                "domain": "academic",
                "question": "What is X?",
                "expected_existence": "answerable",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    m = build_question_domain_map_from_path(path)
    assert m == {"academic_known_real_0000": "academic"}


# ---------- CLI argparse ----------


def test_extract_batch_parse_args_defaults_and_required():
    """CLI must accept the symmetric arg set used by the Step 0 dress
    rehearsal shell pipeline. Defaults match plan §16 (Sonnet 4.6 extractor,
    2048 max_output_tokens)."""
    from src.extract_batch import parse_args

    args = parse_args(
        [
            "--generations",
            "data/generations/pilot_b/claude-haiku-4-5-20251001.jsonl",
            "--questions",
            "data/questions/pilot_15_phase_b.jsonl",
            "--output",
            "data/extractions/pilot_b/claude-haiku-4-5-20251001.jsonl",
            "--manifest",
            "data/extractions/pilot_b/batches.manifest.jsonl",
        ]
    )
    assert args.generations == Path(
        "data/generations/pilot_b/claude-haiku-4-5-20251001.jsonl"
    )
    assert args.questions == Path("data/questions/pilot_15_phase_b.jsonl")
    assert args.output == Path(
        "data/extractions/pilot_b/claude-haiku-4-5-20251001.jsonl"
    )
    assert args.manifest == Path("data/extractions/pilot_b/batches.manifest.jsonl")
    assert args.extractor_model == "claude-sonnet-4-6"
    assert args.max_output_tokens == 2048


def test_extract_batch_parse_args_overrides():
    """--extractor-model and --max-output-tokens are overridable."""
    from src.extract_batch import parse_args

    args = parse_args(
        [
            "--generations",
            "g.jsonl",
            "--questions",
            "q.jsonl",
            "--output",
            "e.jsonl",
            "--manifest",
            "m.jsonl",
            "--extractor-model",
            "claude-haiku-4-5-20251001",
            "--max-output-tokens",
            "1024",
        ]
    )
    assert args.extractor_model == "claude-haiku-4-5-20251001"
    assert args.max_output_tokens == 1024


# ---------- dedup_citations (amendment #6 (b)) ----------
#
# Haiku 4.5 extractor over-counts when the same canonical work is referenced
# in multiple paragraphs of a single response (e.g. 3x "Attention Is All You
# Need" in a length_matched response that opens, explains, and re-cites the
# Transformer paper). Calibration on the 60-gen pilot_b sample showed per-record
# count agreement 86.7% against Sonnet baseline, driven entirely by this failure
# mode. dedup_citations collapses citations sharing normalized title +
# normalized first-author surname + year (within +/-1), preserving the longest
# raw_span. See PRE_REGISTRATION_OSF.md S11 deviation #6.


def test_dedup_citations_empty_list_returns_empty():
    from src.extract_batch import dedup_citations
    assert dedup_citations([]) == []


def test_dedup_citations_single_citation_unchanged():
    from src.extract_batch import dedup_citations
    cite = {"raw_span": "Vaswani et al. (2017)", "title": "Attention Is All You Need",
            "authors": ["Vaswani"], "year": 2017}
    assert dedup_citations([cite]) == [cite]


def test_dedup_citations_collapses_three_attention_entries_preserving_longest_raw_span():
    from src.extract_batch import dedup_citations
    cites = [
        {"raw_span": "Attention Is All You Need",
         "title": "Attention Is All You Need",
         "authors": ["Vaswani", "Shazeer", "Parmar"], "year": 2017},
        {"raw_span": "Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). Attention Is All You Need. NeurIPS.",
         "title": "Attention is all you need",
         "authors": ["Vaswani"], "year": 2017},
        {"raw_span": "Attention Is All You Need (Vaswani et al., 2017)",
         "title": "Attention Is All You Need",
         "authors": ["Vaswani"], "year": 2017},
    ]
    result = dedup_citations(cites)
    assert len(result) == 1
    # Longest raw_span preserved (the full APA-style citation).
    assert result[0]["raw_span"].startswith("Vaswani, A., Shazeer")


def test_dedup_citations_collapses_year_within_plus_minus_one():
    from src.extract_batch import dedup_citations
    cites = [
        {"raw_span": "BERT (2018 preprint)", "title": "BERT",
         "authors": ["Devlin"], "year": 2018},
        {"raw_span": "BERT (2019 NAACL version)", "title": "BERT",
         "authors": ["Devlin"], "year": 2019},
    ]
    result = dedup_citations(cites)
    assert len(result) == 1


def test_dedup_citations_keeps_separate_year_gap_of_two():
    from src.extract_batch import dedup_citations
    cites = [
        {"raw_span": "BERT (2018)", "title": "BERT",
         "authors": ["Devlin"], "year": 2018},
        {"raw_span": "BERT (2020 update)", "title": "BERT",
         "authors": ["Devlin"], "year": 2020},
    ]
    result = dedup_citations(cites)
    assert len(result) == 2


def test_dedup_citations_keeps_separate_when_first_author_differs():
    from src.extract_batch import dedup_citations
    cites = [
        {"raw_span": "Smith 2017 review", "title": "Review of attention mechanisms",
         "authors": ["Smith"], "year": 2017},
        {"raw_span": "Jones 2017 review", "title": "Review of attention mechanisms",
         "authors": ["Jones"], "year": 2017},
    ]
    result = dedup_citations(cites)
    assert len(result) == 2


def test_dedup_citations_normalizes_punctuation_and_case_in_title():
    from src.extract_batch import dedup_citations
    cites = [
        {"raw_span": "ImageNet: A Large-Scale Hierarchical Image Database",
         "title": "ImageNet: A Large-Scale Hierarchical Image Database",
         "authors": ["Deng"], "year": 2009},
        {"raw_span": "ImageNet - a large-scale hierarchical image database (Deng et al.)",
         "title": "ImageNet  a large scale hierarchical image database",
         "authors": ["Deng"], "year": 2009},
    ]
    result = dedup_citations(cites)
    assert len(result) == 1


def test_dedup_citations_collapses_when_no_authors_on_either_using_title_and_year():
    """Clinical trial citations often have empty authors; fall back to (title, year)."""
    from src.extract_batch import dedup_citations
    cites = [
        {"raw_span": "ISIS-2 trial", "title": "Second International Study of Infarct Survival",
         "authors": [], "year": 1988},
        {"raw_span": "ISIS-2 (1988 Lancet)",
         "title": "Second International Study of Infarct Survival",
         "authors": None, "year": 1988},
    ]
    result = dedup_citations(cites)
    assert len(result) == 1


def test_dedup_citations_handles_missing_year():
    """If year is None on both sides, fall back to title + first-author surname only."""
    from src.extract_batch import dedup_citations
    cites = [
        {"raw_span": "Braunwald's Heart Disease", "title": "Braunwald's Heart Disease",
         "authors": ["Braunwald"], "year": None},
        {"raw_span": "Braunwald textbook", "title": "Braunwald's Heart Disease",
         "authors": ["Braunwald"], "year": None},
    ]
    result = dedup_citations(cites)
    assert len(result) == 1


def test_dedup_citations_does_not_cross_collapse_when_one_has_year_other_does_not():
    """A None year is treated as distinct from a known year (conservative)."""
    from src.extract_batch import dedup_citations
    cites = [
        {"raw_span": "Braunwald 2022", "title": "Braunwald's Heart Disease",
         "authors": ["Braunwald"], "year": 2022},
        {"raw_span": "Braunwald textbook", "title": "Braunwald's Heart Disease",
         "authors": ["Braunwald"], "year": None},
    ]
    result = dedup_citations(cites)
    assert len(result) == 2


def test_parse_extraction_results_applies_dedup_at_extraction_time():
    """The integration point: parse_extraction_results returns deduped citations
    so the on-disk JSONL is already clean."""
    from src.extract_batch import parse_extraction_results

    duplicated = [
        {"input": {"citations": [
            {"raw_span": "Attention Is All You Need",
             "title": "Attention Is All You Need",
             "authors": ["Vaswani"], "year": 2017},
            {"raw_span": "Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.",
             "title": "Attention is all you need",
             "authors": ["Vaswani"], "year": 2017},
            {"raw_span": "(Vaswani et al., 2017)",
             "title": "Attention Is All You Need",
             "authors": ["Vaswani"], "year": 2017},
        ]}}
    ]
    blocks = [SimpleNamespace(type="tool_use", name="record_citations",
                              input=duplicated[0]["input"])]
    item = SimpleNamespace(
        custom_id=BatchKey("academic_known_real_0001", "claude-haiku-4-5-20251001", "neutral", 0).encode(),
        result=SimpleNamespace(
            type="succeeded",
            message=SimpleNamespace(content=blocks),
        ),
    )
    out = list(parse_extraction_results(
        [item], extractor_model_id="claude-haiku-4-5-20251001",
        extracted_at="2026-06-07T00:00:00Z",
    ))
    assert len(out) == 1
    # Three identical Attention citations collapsed to one.
    assert len(out[0].citations) == 1
    # Longest raw_span preserved.
    assert out[0].citations[0]["raw_span"].startswith("Vaswani, A. et al.")
