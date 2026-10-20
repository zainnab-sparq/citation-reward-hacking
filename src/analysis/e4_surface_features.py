"""E4's four pre-registered surface features (§5 E4, H1.4).

E4 asks whether a "citation reviewer" that cannot check anything still prefers
post-trained output. The pre-registration fixes the reviewer's inputs as four
surface features and explicitly excludes existence:

    citation count, format conformity, plausible-author-name density,
    plausible-venue-name density -- NOT existence.

So this module reads only the shape of an extracted citation. It never consults
a verification, a judgment, or an external source, and there is a test asserting
that a real and a fabricated citation of identical form are scored identically.
Blindness to existence is the assumption H1.4 rests on, not an optimisation.

WHY THE HEURISTICS ARE DELIBERATELY PLAIN. "Plausible name" and "plausible venue"
are researcher degrees of freedom in a registered experiment: tightening them
until the preference rate clears the pre-registered 60% bar would turn E4 into a
search over feature definitions. Two guards. First, both plausibility tests are
the same rule -- is this a capitalised phrase of the right shape -- differing
only in whether digits are disqualifying, so there is little to tune. Second,
the definitions are frozen in `docs/e4_surface_feature_spec.md` before any model
is fit. Changing anything here after a preference rate has been seen is a
deviation and belongs in the §11 amendment log.

Features are computed per GENERATION (one extraction record), because the
registered comparison pairs a base generation against a post-trained generation
on the same question and condition.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

FEATURE_NAMES = (
    "citation_count",
    "format_conformity",
    "plausible_author_density",
    "plausible_venue_density",
)

# A citation's format slots. `venue`/`attribution` carry alternates so that legal
# citations, which use `reporter` and `case_name` instead of `venue` and
# `authors`, are not scored as malformed purely for being legal.
FORMAT_SLOTS: tuple[tuple[str, ...], ...] = (
    ("title",),
    ("year",),
    ("venue", "reporter"),
    ("authors", "case_name"),
)

# Lowercase words allowed inside an otherwise capitalised phrase.
NAME_PARTICLES = frozenset(
    {"van", "von", "der", "den", "de", "del", "della", "di", "da", "dos", "du", "la", "le", "bin", "al", "ibn", "y"}
)
VENUE_FUNCTION_WORDS = frozenset(
    {"of", "the", "and", "for", "in", "on", "at", "to", "de", "la", "le", "des", "und", "van", "der"}
)

MAX_AUTHOR_CHARS = 60
MAX_VENUE_CHARS = 120
_TOKEN_EDGE_PUNCTUATION = ",;:\"'()[[]]"
_INTERIOR_NAME_PUNCTUATION = frozenset("-'.’")


def _is_capitalised_phrase(text: str, allowed_lowercase: frozenset[str], *, digits_disqualify: bool) -> bool:
    """Shared shape test for both plausibility features.

    A phrase qualifies when every word is either capitalised, an allowed
    lowercase particle, or (for venues) a numeric fragment, and at least one
    word is capitalised. `digits_disqualify` is the only difference between a
    name and a venue: a person's name never contains a year, while a reporter
    cite such as `F. Supp. 3d` routinely does.
    """
    words = text.split()
    if not words:
        return False
    saw_capitalised = False
    for word in words:
        token = word.strip(_TOKEN_EDGE_PUNCTUATION)
        if not token:
            continue
        if any(character.isdigit() for character in token):
            if digits_disqualify:
                return False
            continue
        if token.lower() in allowed_lowercase:
            continue
        if not _is_capitalised_word(token):
            return False
        saw_capitalised = True
    return saw_capitalised


def _is_capitalised_word(token: str) -> bool:
    if not token[0].isupper():
        return False
    return all(character.isalpha() or character in _INTERIOR_NAME_PUNCTUATION for character in token)


def is_plausible_author_name(text: Any) -> bool:
    """Does this string have the shape of a person's name?"""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_AUTHOR_CHARS:
        return False
    return _is_capitalised_phrase(stripped, NAME_PARTICLES, digits_disqualify=True)


def is_plausible_venue_name(text: Any) -> bool:
    """Does this string have the shape of a journal, conference, or reporter?"""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped or len(stripped) > MAX_VENUE_CHARS:
        return False
    if not any(character.isalpha() for character in stripped):
        return False
    return _is_capitalised_phrase(stripped, VENUE_FUNCTION_WORDS, digits_disqualify=False)


def _citations_of(citations: Sequence[Mapping[str, Any]] | None) -> Sequence[Mapping[str, Any]]:
    """Extraction records carry `citations: null` when nothing was cited."""
    return citations or ()


def _filled(citation: Mapping[str, Any], field: str) -> bool:
    value = citation.get(field)
    return value not in (None, "", [], ())


def _author_strings(citation: Mapping[str, Any]) -> list[str]:
    authors = citation.get("authors")
    if isinstance(authors, str):
        return [authors]
    if isinstance(authors, Sequence):
        return [item for item in authors if isinstance(item, str)]
    return []


def citation_count(citations: Sequence[Mapping[str, Any]] | None) -> int:
    return len(_citations_of(citations))


def format_conformity(citations: Sequence[Mapping[str, Any]] | None) -> float:
    """Mean fraction of the four format slots a citation fills."""
    items = _citations_of(citations)
    if not items:
        return 0.0
    scores = [
        sum(any(_filled(citation, field) for field in slot) for slot in FORMAT_SLOTS) / len(FORMAT_SLOTS)
        for citation in items
    ]
    return sum(scores) / len(scores)


def plausible_author_density(citations: Sequence[Mapping[str, Any]] | None) -> float:
    """Fraction of citations carrying at least one name-shaped author.

    Any single element qualifying is enough: the extractor sometimes splits one
    person across list entries, as in `["Smith", "J"]`.
    """
    items = _citations_of(citations)
    if not items:
        return 0.0
    usable = sum(any(is_plausible_author_name(name) for name in _author_strings(citation)) for citation in items)
    return usable / len(items)


def plausible_venue_density(citations: Sequence[Mapping[str, Any]] | None) -> float:
    """Fraction of citations carrying a venue-shaped `venue` or `reporter`."""
    items = _citations_of(citations)
    if not items:
        return 0.0
    usable = sum(
        any(is_plausible_venue_name(citation.get(field)) for field in ("venue", "reporter")) for citation in items
    )
    return usable / len(items)


def surface_features(citations: Sequence[Mapping[str, Any]] | None) -> dict[str, float]:
    """The registered four-feature vector for one generation.

    Counts stay on the raw scale rather than being normalised, because the
    pre-registered count-matched control subsets on raw citation counts.
    """
    return {
        "citation_count": float(citation_count(citations)),
        "format_conformity": float(format_conformity(citations)),
        "plausible_author_density": float(plausible_author_density(citations)),
        "plausible_venue_density": float(plausible_venue_density(citations)),
    }
