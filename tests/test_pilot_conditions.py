"""Contract tests for the four registered E2 prompt conditions.

Pre-registration §5/§6 + plan §16 commit to four conditions:
    neutral, citations_expected, abstention_permitted, length_matched

length_matched is the verbosity-de-confounding control: it asks for a
response length-equivalent to citations_expected WITHOUT mentioning
citations or sources. The citations_expected vs length_matched contrast
isolates 'asked for citations' from 'asked for a longer answer' and is
reported as a secondary de-confounding analysis (not a confirmatory
E2 family member).

These tests are a static contract on `src.pilot.CONDITIONS` and
`src.elicit.ConditionName` — they catch silent drift in the condition
set before any Phase B production spend.
"""
from __future__ import annotations

import re
from typing import get_args

from src.pilot import CONDITIONS
from src.elicit import ConditionName


EXPECTED_CONDITION_NAMES = {
    "neutral",
    "citations_expected",
    "abstention_permitted",
    "length_matched",
}

# De-confounding requirement: length_matched must NOT mention citations,
# sources, or references — that would defeat the verbosity-isolation purpose.
_CITATION_LEXICON = re.compile(r"\b(citation|cite|source|reference|bibliograph)", re.IGNORECASE)


def test_pilot_conditions_has_exactly_four_entries():
    assert len(CONDITIONS) == 4, (
        f"Expected 4 registered E2 conditions per pre-reg §5; got {len(CONDITIONS)}. "
        f"Found names: {[c.name for c in CONDITIONS]}"
    )


def test_pilot_conditions_names_match_pre_registration():
    actual = {c.name for c in CONDITIONS}
    assert actual == EXPECTED_CONDITION_NAMES, (
        f"Condition name set drift. Expected {EXPECTED_CONDITION_NAMES}, got {actual}."
    )


def test_length_matched_template_does_not_mention_citations():
    length_matched = next(c for c in CONDITIONS if c.name == "length_matched")
    match = _CITATION_LEXICON.search(length_matched.template)
    assert match is None, (
        "length_matched template must NOT mention citation/source/reference vocabulary "
        f"(would defeat the de-confounding contrast). Offending match: {match!r} in "
        f"template: {length_matched.template!r}"
    )


def test_length_matched_template_asks_for_multi_paragraph():
    length_matched = next(c for c in CONDITIONS if c.name == "length_matched")
    template = length_matched.template.lower()
    assert "paragraph" in template, (
        "length_matched must request a multi-paragraph response (length parity with "
        f"citations_expected). Got template: {length_matched.template!r}"
    )


def test_condition_name_literal_includes_length_matched():
    assert "length_matched" in get_args(ConditionName), (
        f"src.elicit.ConditionName Literal must include 'length_matched'. "
        f"Got: {get_args(ConditionName)}"
    )
