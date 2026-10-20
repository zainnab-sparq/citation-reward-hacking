"""Tests for locating each citation inside a generation.

H1.3 classifies *citations*, not generations, so the probe reads hidden states
over the tokens of one citation. Getting from a citation's raw text to a token
range crosses two boundaries where errors are silent:

  - the same citation text can appear more than once in one answer, and one
    citation's text can be a substring of another's (a short-form cite inside a
    long-form one), so naive `str.find` assigns the wrong occurrence
  - character offsets have to become token indices, and an off-by-one there
    pools a neighboring token instead

Neither shows up as an exception. Both show up as a degraded AUROC that looks
like a null result.
"""

import pytest

from src.e3.citation_spans import (
    char_span_to_token_span,
    locate_raw_spans,
)


def test_locates_a_single_citation():
    text = "As shown in Smith (2020), the effect holds."
    assert locate_raw_spans(text, ["Smith (2020)"]) == [(12, 24)]


def test_locates_citations_in_document_order():
    text = "First Alpha (1999) then Beta (2001) end."
    assert locate_raw_spans(text, ["Alpha (1999)", "Beta (2001)"]) == [(6, 18), (24, 35)]


def test_repeated_identical_citations_get_distinct_spans():
    # Two citations with identical text must not both resolve to the first hit,
    # or one activation vector is used twice under two different labels.
    text = "See Ref (1) and again Ref (1) here."
    spans = locate_raw_spans(text, ["Ref (1)", "Ref (1)"])
    assert spans == [(4, 11), (22, 29)]
    assert spans[0] != spans[1]


def test_short_form_citation_nested_in_a_long_form_one_is_not_double_counted():
    # The extractor emits both "Polaris v. Samsung" and the full cite that
    # contains it. Greedy in-order matching must not hand back overlapping spans.
    text = "In Polaris v. Samsung, 386 F. Supp. 3d 760 (N.D. Tex. 2019), the court held."
    long_form = "Polaris v. Samsung, 386 F. Supp. 3d 760 (N.D. Tex. 2019)"
    short_form = "Polaris v. Samsung"
    spans = locate_raw_spans(text, [long_form, short_form])
    assert spans[0] is not None
    # The short form only occurs inside the long form, so there is no second
    # non-overlapping home for it.
    assert spans[1] is None


def test_unlocatable_citation_returns_none_rather_than_guessing():
    text = "No citation here at all."
    assert locate_raw_spans(text, ["Ghost (1999)"]) == [None]


def test_whitespace_differences_do_not_defeat_the_match():
    text = "See  Smith   (2020)  for details."
    spans = locate_raw_spans(text, ["Smith (2020)"])
    assert spans[0] is not None
    start, end = spans[0]
    assert "Smith" in text[start:end] and "2020" in text[start:end]


def test_char_span_maps_to_the_tokens_that_cover_it():
    # offset_mapping as a tokenizer returns it: (char_start, char_end) per token.
    offsets = [(0, 3), (3, 7), (7, 12), (12, 18), (18, 22)]
    assert char_span_to_token_span(offsets, (7, 18)) == (2, 4)


def test_char_span_includes_a_partially_covered_token():
    offsets = [(0, 5), (5, 10), (10, 15)]
    # Chars 7..12 straddle tokens 1 and 2; both must be included.
    assert char_span_to_token_span(offsets, (7, 12)) == (1, 3)


def test_char_span_ignores_zero_width_special_tokens():
    # Special tokens report (0, 0) and would otherwise be swept into every span.
    offsets = [(0, 0), (0, 4), (4, 9), (0, 0)]
    assert char_span_to_token_span(offsets, (0, 9)) == (1, 3)


def test_char_span_with_no_overlapping_token_fails_loudly():
    offsets = [(0, 5), (5, 10)]
    with pytest.raises(ValueError, match="no tokens"):
        char_span_to_token_span(offsets, (50, 60))
