"""Locate each extracted citation inside its generation, then in token space.

The E3 probe reads hidden states over the tokens of one citation, because H1.3
classifies citations rather than generations. That requires mapping a citation's
raw text to a character range in `output_text`, then to token indices.

Matching is greedy and in document order, with a cursor that advances past each
assigned span. Two properties of the extractor's output force that:

  - the same citation can legitimately appear twice in one answer, and both
    occurrences carry their own verdict, so both need their own span
  - a short-form cite is often a literal substring of the long-form cite in the
    same answer, so a plain search would hand back a span already claimed

A citation that cannot be placed returns None rather than a guess. Those are
counted and reported, never silently dropped, because a systematic failure to
locate one kind of citation would bias the probe's training set.
"""

from __future__ import annotations

import re
from typing import Sequence

# Collapse runs of whitespace when the literal match fails. Model output and the
# extractor's raw_span sometimes disagree on internal spacing and line breaks.
_WS = re.compile(r"\s+")


def _flexible_pattern(raw: str) -> re.Pattern:
    return re.compile(r"\s+".join(re.escape(part) for part in _WS.split(raw.strip())))


def locate_raw_spans(
    text: str, raw_spans: Sequence[str]
) -> list[tuple[int, int] | None]:
    """Character spans for each citation, in the order the extractor emitted them.

    Returns None for any citation with no unclaimed occurrence.
    """
    claimed: list[tuple[int, int]] = []
    out: list[tuple[int, int] | None] = []

    def _overlaps(start: int, end: int) -> bool:
        return any(start < c_end and c_start < end for c_start, c_end in claimed)

    for raw in raw_spans:
        if not raw or not raw.strip():
            out.append(None)
            continue

        found = None
        for pattern in (re.compile(re.escape(raw)), _flexible_pattern(raw)):
            for match in pattern.finditer(text):
                if not _overlaps(match.start(), match.end()):
                    found = (match.start(), match.end())
                    break
            if found:
                break

        out.append(found)
        if found:
            claimed.append(found)
    return out


def char_span_to_token_span(
    offset_mapping: Sequence[tuple[int, int]], char_span: tuple[int, int]
) -> tuple[int, int]:
    """Half-open token range covering a character range.

    Any token that overlaps the character range at all is included, so a token
    straddling the boundary is kept rather than dropped. Zero-width entries are
    special tokens and are excluded; a tokenizer reports those as (0, 0), which
    would otherwise land inside every span that starts at character zero.
    """
    char_start, char_end = char_span
    indices = [
        i
        for i, (tok_start, tok_end) in enumerate(offset_mapping)
        if tok_end > tok_start and tok_start < char_end and char_start < tok_end
    ]
    if not indices:
        raise ValueError(
            f"no tokens overlap character span {char_span}; "
            "the citation cannot be placed in token space"
        )
    return (indices[0], indices[-1] + 1)
