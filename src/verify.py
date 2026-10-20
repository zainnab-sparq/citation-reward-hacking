"""Citation verification pipeline.

Parses citations from raw text and verifies them against external sources
(Crossref, Semantic Scholar, OpenAlex, CourtListener).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import httpx
from rapidfuzz import fuzz

Domain = Literal["academic", "medical", "legal"]

_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>)]+", re.IGNORECASE)
_YEAR_RE = re.compile(r"\((\d{4})[a-z]?\)")
_APA_AFTER_YEAR_RE = re.compile(r"\(\d{4}[a-z]?\)\.\s*([^.]+?)\.\s")
_FIRST_AUTHOR_SURNAME_RE = re.compile(r"^\s*([A-Z][A-Za-z'\-]+)\s*,")
_PUNCT_RE = re.compile(r"[^\w\s]")

# Legal-domain patterns. case_name = "Plaintiff v. Defendant", anchored on a
# capitalized first letter on each side of " v. " and bounded by comma or paren.
# reporter = "<vol> <abbrev> <page>", e.g. "384 U.S. 436" or "123 F.3d 456".
# legal_year captures a year inside parens, allowing jurisdiction prefix like
# "(9th Cir. 2001)" — academic _YEAR_RE requires the strict "(YYYY)" form.
_CASE_NAME_RE = re.compile(r"\b([A-Z][^,()]*?\s+v\.\s+[A-Z][^,()]*?)(?=\s*,|\s*\(|$)")
_REPORTER_RE = re.compile(r"\b(\d+\s+[A-Z][\w.]*?\s+\d+)\b")
_LEGAL_YEAR_RE = re.compile(r"\((?:[^)]*?\b)?(\d{4})\b[^)]*\)")

# Court canonicalization: collapse the many spellings of US federal court
# identifiers into a stable key so cited-vs-candidate court alignment is a
# string comparison. Returns None when no recognizable court signal is present
# (caller treats unknown as "do not gate on court"). Patterns are intentionally
# narrow — they only match the circuits + SCOTUS, which is the Phase B legal
# scope per pre-reg §15 (US federal case law only).
_ORDINAL_CIRCUIT_WORD = {
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5",
    "sixth": "6", "seventh": "7", "eighth": "8", "ninth": "9", "tenth": "10",
    "eleventh": "11", "twelfth": "12", "thirteenth": "13",
}
_NUMERIC_CIRCUIT_RE = re.compile(
    r"\b(\d+)\s*(?:st|nd|rd|th)\s+cir(?:\.|cuit)?\b", re.IGNORECASE
)
_WORD_CIRCUIT_RE = re.compile(
    r"\b(" + "|".join(_ORDINAL_CIRCUIT_WORD) + r")\s+circuit\b", re.IGNORECASE
)
_DC_CIRCUIT_RE = re.compile(r"\bd\.?\s*c\.?\s+cir(?:\.|cuit)?\b", re.IGNORECASE)
_FEDERAL_CIRCUIT_RE = re.compile(
    r"\bfed(?:eral)?\.?\s+cir(?:\.|cuit)?\b", re.IGNORECASE
)
_SCOTUS_RE = re.compile(
    r"\bsupreme\s+court(?:\s+of\s+the\s+united\s+states)?\b|\bscotus\b",
    re.IGNORECASE,
)


def _canonical_court(text: str | None) -> str | None:
    """Map a free-text court reference to a canonical key.

    Returns one of: "scotus", "circuit:N" (N=1..13), "circuit:dc",
    "circuit:federal", or None if no recognizable signal is present.
    The empty string and None both yield None so callers can pass raw API
    payload fields without guarding.
    """
    if not text:
        return None
    if _SCOTUS_RE.search(text):
        return "scotus"
    m = _DC_CIRCUIT_RE.search(text)
    if m:
        return "circuit:dc"
    m = _FEDERAL_CIRCUIT_RE.search(text)
    if m:
        return "circuit:federal"
    m = _NUMERIC_CIRCUIT_RE.search(text)
    if m:
        return f"circuit:{int(m.group(1))}"
    m = _WORD_CIRCUIT_RE.search(text)
    if m:
        return f"circuit:{_ORDINAL_CIRCUIT_WORD[m.group(1).lower()]}"
    return None


_CL_YEAR_TOLERANCE = 1  # ±1 year (publication-vs-decision lag is common)

# Westlaw and LexisNexis citation formats: "YYYY WL <number>", "YYYY U.S. Dist.
# LEXIS <number>", etc. These are proprietary databases not indexed in
# CourtListener's free opinion search, so including them in CL queries returns
# 0 results even for real cases. Step 0 hand-check surfaced this on Mata v.
# Avianca (cited as "2023 WL 4015181"). Pre-reg §11 amendment #4.
_WL_LEXIS_REPORTER_RE = re.compile(r"\b(WL|LEXIS)\b", re.IGNORECASE)

# Loose caption-match threshold for distinguishing "CL found same-caption cases"
# (the cited case doesn't exist at the cited year/court — definitive negative) from
# "CL search returned body-mentions of the cited case in unrelated opinions" (CL
# coverage gap — non-definitive). Tuned with Step 0 examples: a real-but-CL-absent
# case (Mata v. Avianca) returns candidates with ~10–20 token_set_ratio to the
# cited name; a fictitious-but-common-surname case (United States v. Chen) returns
# candidates with 100 (exact caption match). 60 sits comfortably between.
_CL_LOOSE_CAPTION_THRESHOLD = 60

# Title-search widen: top-5 surfaced derivative works (FAD-BERT, Sentence-BERT)
# above the original BERT NAACL paper, so the verifier never saw the real paper.
# Top-25 puts most well-cited originals in scope while keeping the per-citation
# API cost flat (Crossref / S2 / OpenAlex all return up to 25 per request).
_TITLE_SEARCH_ROWS = 25

# Year tolerance for academic title-search candidate alignment. Same rationale as
# _CL_YEAR_TOLERANCE: arXiv-vs-conference publication lag is commonly 1 year.
_TITLE_YEAR_TOLERANCE = 1


def _first_surname(authors: tuple[str, ...]) -> str | None:
    """Extract a clean lowercase surname from the first author entry.

    Handles 'Devlin, J.' (APA), 'Devlin J' (Vancouver), and bare 'Devlin' forms.
    Returns None when authors is empty or the first entry has no recoverable
    surname (caller treats unknown as "do not gate on author").
    """
    if not authors:
        return None
    first = authors[0].strip()
    if not first:
        return None
    if "," in first:
        first = first.split(",", 1)[0]
    # Take the leading word as the surname (drop initials / given names).
    for token in first.split():
        clean = re.sub(r"[^A-Za-z'\-]", "", token).strip()
        if len(clean) >= 2:
            return clean.lower()
    return None


def _candidate_publication_year(item: dict[str, Any]) -> int | None:
    """Pull a publication year from a Crossref / S2 / OpenAlex result item.

    Crossref uses {"issued": {"date-parts": [[YYYY, MM, DD]]}}.
    S2 uses {"year": YYYY}.
    OpenAlex uses {"publication_year": YYYY}.
    """
    issued = item.get("issued")
    if isinstance(issued, dict):
        parts = issued.get("date-parts")
        if isinstance(parts, list) and parts and isinstance(parts[0], list) and parts[0]:
            head = parts[0][0]
            if isinstance(head, int):
                return head
            if isinstance(head, str) and head.isdigit():
                return int(head)
    for key in ("year", "publication_year"):
        v = item.get(key)
        if isinstance(v, int):
            return v
        if isinstance(v, str) and v.isdigit():
            return int(v)
    return None


def _candidate_first_surname(item: dict[str, Any]) -> str | None:
    """Pull the first-author surname from a Crossref / S2 / OpenAlex result item."""
    # Crossref: {"author": [{"family": "Devlin", "given": "Jacob"}, ...]}
    crossref_authors = item.get("author")
    if isinstance(crossref_authors, list) and crossref_authors:
        first = crossref_authors[0]
        if isinstance(first, dict):
            family = first.get("family")
            if isinstance(family, str) and family:
                return family.strip().lower()
    # S2: {"authors": [{"name": "Jacob Devlin"}, ...]}  (surname last)
    s2_authors = item.get("authors")
    if isinstance(s2_authors, list) and s2_authors:
        first = s2_authors[0]
        if isinstance(first, dict):
            name = first.get("name")
            if isinstance(name, str) and name.strip():
                tokens = name.strip().split()
                if tokens:
                    return tokens[-1].strip(",.").lower()
    # OpenAlex: {"authorships": [{"author": {"display_name": "Jacob Devlin"}}, ...]}
    oa_authorships = item.get("authorships")
    if isinstance(oa_authorships, list) and oa_authorships:
        first = oa_authorships[0]
        if isinstance(first, dict):
            author = first.get("author")
            if isinstance(author, dict):
                name = author.get("display_name")
                if isinstance(name, str) and name.strip():
                    tokens = name.strip().split()
                    if tokens:
                        return tokens[-1].strip(",.").lower()
    return None


def _candidate_year_aligned_acad(item: dict[str, Any], cited_year: int | None) -> bool:
    """True if the candidate's publication year is within ±tolerance of cited_year.

    Unknown on either side → True (don't gate on year we couldn't recover).
    """
    if cited_year is None:
        return True
    cy = _candidate_publication_year(item)
    if cy is None:
        return True
    return abs(cy - cited_year) <= _TITLE_YEAR_TOLERANCE


def _candidate_surname_aligned(
    item: dict[str, Any], cited_first_surname: str | None
) -> bool:
    """True if the candidate's first-author surname matches the cited first surname.

    Unknown on either side → True.
    """
    if not cited_first_surname:
        return True
    cs = _candidate_first_surname(item)
    if not cs:
        return True
    return cs == cited_first_surname


def _normalize_title(title: str) -> str:
    """Lowercase + strip punctuation for stable fuzzy matching."""
    return _PUNCT_RE.sub(" ", title.lower()).strip()


def _title_matches(query: str, candidate: str) -> bool:
    """A title match requires BOTH:
    - token_set_ratio >= 95 (token overlap, robust to small wording variation)
    - token_sort_ratio >= 85 (preserves word order, prevents subset over-matching)

    Using token_set_ratio alone is too permissive on short queries: e.g., a query
    "Imaginary paper" matches a candidate "Review Paper on Imaginary Numbers"
    at 100, and a query "Quantum entanglement of ... North Dakota" matches a
    candidate just titled "North Dakota" at 100. Requiring both ratios above
    threshold blocks both pathologies while still tolerating real noise like
    title-case variation or trailing punctuation differences.
    """
    q = _normalize_title(query)
    c = _normalize_title(candidate)
    return fuzz.token_set_ratio(q, c) >= _FUZZY_MATCH and fuzz.token_sort_ratio(q, c) >= _FUZZY_CANDIDATE
_CROSSREF_BASE = "https://api.crossref.org/works"
_S2_BASE = "https://api.semanticscholar.org/graph/v1/paper"
_OPENALEX_BASE = "https://api.openalex.org/works"
_COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v4/search/"
_COURTLISTENER_TOKEN_ENV = "COURTLISTENER_API_TOKEN"
_S2_API_KEY_ENV = "SEMANTIC_SCHOLAR_API_KEY"


def _s2_headers() -> dict[str, str]:
    """S2 anonymous tier hits 429 quickly on bulk runs and false-fabricates
    arXiv papers. Free API key from semanticscholar.org/product/api lifts limits.
    """
    key = os.environ.get(_S2_API_KEY_ENV)
    return {"x-api-key": key} if key else {}
_DEFAULT_TIMEOUT = httpx.Timeout(10.0)

# Field-standard thresholds (Vikranth pipeline; consistent with HalluLens/CiteLab norms):
# token_set_ratio >= 95 = match, 85-94 = candidate (flag for human review),
# < 85 = no match.
_FUZZY_MATCH = 95
_FUZZY_CANDIDATE = 85


class VerificationStatus(StrEnum):
    REAL = "REAL"
    FABRICATED = "FABRICATED"
    UNVERIFIABLE = "UNVERIFIABLE"


@dataclass(frozen=True)
class Citation:
    raw_text: str
    domain: Domain
    title: str | None = None
    authors: tuple[str, ...] = field(default_factory=tuple)
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    case_name: str | None = None
    reporter: str | None = None


@dataclass
class VerificationResult:
    citation: Citation
    status: VerificationStatus
    sources_checked: list[str] = field(default_factory=list)
    matches: dict[str, dict[str, Any]] = field(default_factory=dict)
    confidence: float = 0.0
    notes: str = ""

    def to_cache_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "sources_checked": self.sources_checked,
            "matches": self.matches,
            "confidence": self.confidence,
            "notes": self.notes,
        }


class VerificationCache:
    """Append-only JSONL cache keyed by (raw_text, domain).

    Loads existing entries on construction. Writes are buffered and flushed
    explicitly via flush(); set() also appends to disk for crash safety.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._store: dict[str, dict[str, Any]] = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                self._store[record["key"]] = record["value"]

    def get(self, key: str) -> dict[str, Any] | None:
        return self._store.get(key)

    def set(self, key: str, value: dict[str, Any]) -> None:
        self._store[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"key": key, "value": value}) + "\n")

    def flush(self) -> None:
        # set() already writes through; flush is a no-op for the append-only design.
        pass

    @staticmethod
    def make_key(raw_text: str, domain: str) -> str:
        digest = hashlib.sha256(f"{domain}::{raw_text}".encode()).hexdigest()
        return digest[:32]


def parse_citation(text: str, domain: Domain) -> Citation:
    doi_match = _DOI_RE.search(text)
    doi = doi_match.group(0).rstrip(".,;") if doi_match else None

    year_match = _YEAR_RE.search(text)
    year = int(year_match.group(1)) if year_match else None

    title_match = _APA_AFTER_YEAR_RE.search(text)
    title = title_match.group(1).strip() if title_match else None

    authors: tuple[str, ...] = ()
    surname_match = _FIRST_AUTHOR_SURNAME_RE.match(text)
    if surname_match:
        authors = (surname_match.group(1),)

    case_name: str | None = None
    reporter: str | None = None
    if domain == "legal":
        case_match = _CASE_NAME_RE.search(text)
        if case_match:
            case_name = case_match.group(1).strip()
        reporter_match = _REPORTER_RE.search(text)
        if reporter_match:
            reporter = reporter_match.group(1).strip()
        # Legal year may include jurisdiction prefix: "(9th Cir. 2001)"
        if year is None:
            legal_year_match = _LEGAL_YEAR_RE.search(text)
            if legal_year_match:
                year = int(legal_year_match.group(1))

    return Citation(
        raw_text=text,
        domain=domain,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        case_name=case_name,
        reporter=reporter,
    )


@dataclass(frozen=True)
class LookupOutcome:
    """Result of a single source lookup. Distinguishes definitive misses from errors."""
    found: bool
    definitive: bool  # True if the source authoritatively says no-such-record (e.g., 404)
    payload: dict[str, Any] | None = None


async def _lookup_crossref_doi(doi: str, client: httpx.AsyncClient) -> LookupOutcome:
    """Look up a DOI in Crossref.

    A 200 with a payload is a definitive positive (REAL).
    A 404 is NOT definitive — Crossref does not index arXiv (DataCite),
    Zenodo, figshare, or many other DOI registries. We fall through to
    title-based search and let that determine FABRICATED.
    """
    try:
        response = await client.get(f"{_CROSSREF_BASE}/{doi}")
    except httpx.HTTPError:
        return LookupOutcome(found=False, definitive=False)
    if response.status_code == 200:
        message = response.json().get("message")
        return LookupOutcome(found=True, definitive=True, payload=message)
    return LookupOutcome(found=False, definitive=False)


_ARXIV_DOI_RE = re.compile(r"^10\.48550/arXiv\.(\d{4}\.\d{4,5})$", re.IGNORECASE)


def _s2_identifier_for_doi(doi: str) -> str:
    """S2 indexes arXiv papers under `arXiv:<id>`, not `DOI:10.48550/arXiv.<id>`.

    Without this rewrite, every arXiv DOI returned 404 and got classified as
    FABRICATED — the BERT paper among them. Caught by the Phase A.5 re-pilot.
    """
    arxiv_match = _ARXIV_DOI_RE.match(doi)
    if arxiv_match:
        return f"arXiv:{arxiv_match.group(1)}"
    return f"DOI:{doi}"


async def _lookup_s2_doi(doi: str, client: httpx.AsyncClient) -> LookupOutcome:
    """Look up a DOI in Semantic Scholar. Covers arXiv preprints (Crossref does not)."""
    identifier = _s2_identifier_for_doi(doi)
    try:
        response = await client.get(
            f"{_S2_BASE}/{identifier}",
            params={"fields": "title,year,authors"},
            headers=_s2_headers(),
        )
    except httpx.HTTPError:
        return LookupOutcome(found=False, definitive=False)
    if response.status_code == 200:
        return LookupOutcome(found=True, definitive=True, payload=response.json())
    if response.status_code == 404:
        return LookupOutcome(found=False, definitive=True)
    return LookupOutcome(found=False, definitive=False)


async def _lookup_openalex_doi(doi: str, client: httpx.AsyncClient) -> LookupOutcome:
    """Look up a DOI in OpenAlex. Broadest coverage of free academic indexes."""
    try:
        response = await client.get(f"{_OPENALEX_BASE}/https://doi.org/{doi}")
    except httpx.HTTPError:
        return LookupOutcome(found=False, definitive=False)
    if response.status_code == 200:
        return LookupOutcome(found=True, definitive=True, payload=response.json())
    if response.status_code == 404:
        return LookupOutcome(found=False, definitive=True)
    return LookupOutcome(found=False, definitive=False)


async def _lookup_openalex_title(
    title: str,
    client: httpx.AsyncClient,
    rows: int = _TITLE_SEARCH_ROWS,
    *,
    cited_year: int | None = None,
    cited_first_surname: str | None = None,
) -> LookupOutcome:
    # Step 0 hand-check found that OpenAlex title search returns derivative
    # works (Sentence-BERT, TinyBERT, ...) above the original BERT NAACL paper
    # — even at top 25, the original never appears. When the cited first-author
    # surname is known, constrain the search to candidates by that author so
    # the original surfaces. Pre-reg §11 amendment #4.
    params: dict[str, str | int] = {"search": title, "per-page": rows}
    if cited_first_surname:
        params["filter"] = (
            f"authorships.author.display_name.search:{cited_first_surname}"
        )
    try:
        response = await client.get(_OPENALEX_BASE, params=params)
    except httpx.HTTPError:
        return LookupOutcome(found=False, definitive=False)
    if response.status_code != 200:
        return LookupOutcome(found=False, definitive=False)
    items = response.json().get("results", []) or []
    for item in items:
        candidate = item.get("title") or item.get("display_name") or ""
        if not _title_matches(title, candidate):
            continue
        if not _candidate_year_aligned_acad(item, cited_year):
            continue
        if not _candidate_surname_aligned(item, cited_first_surname):
            continue
        return LookupOutcome(found=True, definitive=True, payload=item)
    return LookupOutcome(found=False, definitive=True)


async def _lookup_s2_title(
    title: str,
    client: httpx.AsyncClient,
    rows: int = _TITLE_SEARCH_ROWS,
    *,
    cited_year: int | None = None,
    cited_first_surname: str | None = None,
) -> LookupOutcome:
    # S2's `query` is a keyword search across title + author fields; including
    # the cited first-author surname dramatically improves precision when the
    # title is shared by many derivative works (BERT, ResNet, etc.).
    query = title if not cited_first_surname else f"{title} {cited_first_surname}"
    try:
        response = await client.get(
            f"{_S2_BASE}/search",
            params={"query": query, "limit": rows, "fields": "title,year,authors"},
            headers=_s2_headers(),
        )
    except httpx.HTTPError:
        return LookupOutcome(found=False, definitive=False)
    if response.status_code != 200:
        return LookupOutcome(found=False, definitive=False)
    items = response.json().get("data", []) or []
    for item in items:
        candidate = item.get("title") or ""
        if not _title_matches(title, candidate):
            continue
        if not _candidate_year_aligned_acad(item, cited_year):
            continue
        if not _candidate_surname_aligned(item, cited_first_surname):
            continue
        return LookupOutcome(found=True, definitive=True, payload=item)
    return LookupOutcome(found=False, definitive=True)


async def _lookup_crossref_title(
    title: str,
    client: httpx.AsyncClient,
    rows: int = _TITLE_SEARCH_ROWS,
    *,
    cited_year: int | None = None,
    cited_first_surname: str | None = None,
) -> LookupOutcome:
    # Crossref's `query.title` + `query.author` are intersected. When cited
    # author surname is known, this is what reliably surfaces the original
    # paper above derivative works whose titles share heavy token overlap
    # ("FAD-BERT", "Sentence-BERT", ...). Pre-reg §11 amendment #4.
    params: dict[str, str | int] = {"query.title": title, "rows": rows}
    if cited_first_surname:
        params["query.author"] = cited_first_surname
    try:
        response = await client.get(_CROSSREF_BASE, params=params)
    except httpx.HTTPError:
        return LookupOutcome(found=False, definitive=False)
    if response.status_code != 200:
        return LookupOutcome(found=False, definitive=False)
    items = response.json().get("message", {}).get("items", [])
    for item in items:
        candidate_titles = item.get("title", []) or []
        title_matched = any(_title_matches(title, c) for c in candidate_titles)
        if not title_matched:
            continue
        if not _candidate_year_aligned_acad(item, cited_year):
            continue
        if not _candidate_surname_aligned(item, cited_first_surname):
            continue
        return LookupOutcome(found=True, definitive=True, payload=item)
    # API returned but no items matched — definitive negative
    return LookupOutcome(found=False, definitive=True)


async def _lookup_courtlistener(
    citation: Citation,
    client: httpx.AsyncClient,
) -> LookupOutcome:
    """Search CourtListener (v4 search API) for a US case citation.

    Crossref / S2 / OpenAlex are academic-paper indexes; they do not cover US
    case law. CourtListener does. The Phase A pilot exposed this gap by marking
    real Miranda v. Arizona and Loving v. Virginia as fake.

    Auth: CourtListener v4 requires a free API token. Set COURTLISTENER_API_TOKEN
    in env. If missing, returns non-definitive (UNVERIFIABLE), not FABRICATED —
    so we never falsely fabricate a real case just because the token wasn't set.

    Strategy: build a quoted-string query from reporter and/or case_name and
    issue a single search. A quoted reporter is uniquely identifying; a
    case_name needs fuzzy verification against the returned caseName field.
    """
    # Step 0 hand-check (§11 amendment #4): Westlaw / Lexis citations are not
    # tokenized in CL's full-text search index. Including them in the query
    # returns 0 even for real cases (e.g. Mata v. Avianca, 2023 WL 4015181).
    # Drop them from the query so the case_name path is exercised.
    reporter_for_query: str | None = None
    if citation.reporter and not _WL_LEXIS_REPORTER_RE.search(citation.reporter):
        reporter_for_query = citation.reporter

    query_parts: list[str] = []
    if reporter_for_query:
        query_parts.append(f'"{reporter_for_query}"')
    if citation.case_name:
        query_parts.append(f'"{citation.case_name}"')
    if not query_parts:
        return LookupOutcome(found=False, definitive=False)

    token = os.environ.get(_COURTLISTENER_TOKEN_ENV)
    headers = {"Authorization": f"Token {token}"} if token else {}

    try:
        response = await client.get(
            _COURTLISTENER_BASE,
            params={"q": " ".join(query_parts), "type": "o"},
            headers=headers,
        )
    except httpx.HTTPError:
        return LookupOutcome(found=False, definitive=False)
    # 401/403 = no/bad token — not a definitive negative for the citation itself
    if response.status_code in (401, 403):
        return LookupOutcome(found=False, definitive=False)
    if response.status_code != 200:
        return LookupOutcome(found=False, definitive=False)

    results = response.json().get("results", []) or []
    if not results:
        return LookupOutcome(found=False, definitive=True)

    if citation.case_name:
        # Step 0 dress rehearsal surfaced a name-only false positive on
        # common-surname unanswerables ("United States v. Chen" matched a real
        # First-Circuit 2021 case despite the cited 11th Cir. 2024 venue). We
        # now also require year-alignment and court-alignment when those
        # signals are present on both sides. Either signal being None on the
        # cited or candidate side falls back to the prior behavior (skip that
        # check) — see pre-reg §11 amendment #3.
        #
        # The Step 0 hand-check surfaced a second failure mode (pre-reg §11
        # amendment #4): CL's v4 search is full-text body matching, not
        # caption-only, so a query for a real-but-CL-absent case (Mata v.
        # Avianca, SDNY 2023 docket order outside CL's opinion index)
        # returns N>0 candidates that merely *mention* the cited case in
        # their text. We distinguish that "coverage gap" from a "fictitious-
        # case" definitive negative by tracking whether ANY candidate's
        # caseName even loosely shares the cited caption tokens. If none
        # share caption tokens, the result is non-definitive.
        cited_court = _canonical_court(citation.venue) or _canonical_court(
            citation.raw_text
        )
        cited_case_norm = _normalize_title(citation.case_name)
        any_caption_match = False
        for result in results:
            candidate = result.get("caseName") or ""
            strict_caption = _title_matches(citation.case_name, candidate)
            if strict_caption:
                any_caption_match = True
            elif (
                fuzz.token_set_ratio(cited_case_norm, _normalize_title(candidate))
                >= _CL_LOOSE_CAPTION_THRESHOLD
            ):
                any_caption_match = True
            if not strict_caption:
                continue
            if not _candidate_year_aligned(result, citation.year):
                continue
            if not _candidate_court_aligned(result, cited_court):
                continue
            return LookupOutcome(found=True, definitive=True, payload=result)
        return LookupOutcome(found=False, definitive=any_caption_match)

    # No case_name available — trust the reporter-based query
    return LookupOutcome(found=True, definitive=True, payload=results[0])


def _candidate_year_aligned(result: dict[str, Any], cited_year: int | None) -> bool:
    """True if the candidate's dateFiled year is within ±tolerance of cited_year.

    Unknown cited year → True (don't gate on year we couldn't recover).
    Unknown candidate dateFiled → True (don't punish a valid match because the
    CL row lacks the field).
    """
    if cited_year is None:
        return True
    date_filed = result.get("dateFiled") or ""
    if len(date_filed) < 4 or not date_filed[:4].isdigit():
        return True
    candidate_year = int(date_filed[:4])
    return abs(candidate_year - cited_year) <= _CL_YEAR_TOLERANCE


def _candidate_court_aligned(
    result: dict[str, Any], cited_court: str | None
) -> bool:
    """True if the candidate's court canonicalizes to the same key as cited_court.

    Unknown cited court → True. Unknown candidate court → True.
    """
    if cited_court is None:
        return True
    candidate_court = _canonical_court(result.get("court"))
    if candidate_court is None:
        return True
    return candidate_court == cited_court


async def verify_citation(
    text: str,
    domain: Domain,
    client: httpx.AsyncClient | None = None,
    cache: VerificationCache | None = None,
    *,
    parsed: Citation | None = None,
) -> VerificationResult:
    """Verify a citation. If `parsed` is provided, use it instead of regex parsing.

    The LLM extractor (src/extract_citations.py) produces structured citations
    with cleaner title / author / venue fields than the regex parser can extract
    from messy raw spans. Pass it via `parsed=` to preserve those fields.
    """
    citation = parsed if parsed is not None else parse_citation(text, domain=domain)

    cache_key = VerificationCache.make_key(text, domain) if cache else None
    if cache and cache_key:
        cached = cache.get(cache_key)
        if cached is not None:
            return VerificationResult(
                citation=citation,
                status=VerificationStatus(cached["status"]),
                sources_checked=list(cached.get("sources_checked", [])),
                matches=dict(cached.get("matches", {})),
                confidence=float(cached.get("confidence", 0.0)),
                notes=str(cached.get("notes", "")),
            )

    own_client = client is None
    if own_client:
        client = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
    try:
        result = VerificationResult(citation=citation, status=VerificationStatus.UNVERIFIABLE)
        resolved = False  # set to True once a positive (REAL) match is found
        # Track whether each source's negative result was definitive — needed to
        # distinguish FABRICATED (multiple sources said no) from UNVERIFIABLE
        # (network failures or coverage gaps).
        negative_definitive_count = 0
        sources_attempted = 0

        async def _try_doi_source(name: str, lookup) -> bool:
            nonlocal resolved, sources_attempted, negative_definitive_count
            if resolved:
                return False
            sources_attempted += 1
            result.sources_checked.append(name)
            outcome = await lookup(citation.doi, client)
            if outcome.found:
                result.matches[name] = outcome.payload or {}
                result.status = VerificationStatus.REAL
                result.confidence = 1.0
                resolved = True
                return True
            if outcome.definitive:
                negative_definitive_count += 1
            return False

        cited_first_surname = _first_surname(citation.authors)

        async def _try_title_source(name: str, lookup) -> bool:
            nonlocal resolved, sources_attempted, negative_definitive_count
            if resolved:
                return False
            sources_attempted += 1
            result.sources_checked.append(name)
            outcome = await lookup(
                citation.title,
                client,
                cited_year=citation.year,
                cited_first_surname=cited_first_surname,
            )
            if outcome.found:
                result.matches[name] = outcome.payload or {}
                result.status = VerificationStatus.REAL
                result.confidence = 0.85
                resolved = True
                return True
            if outcome.definitive:
                negative_definitive_count += 1
            return False

        # ---- Legal path: CourtListener (US case law not in academic indexes) ----
        if domain == "legal" and (citation.case_name or citation.reporter):
            sources_attempted += 1
            result.sources_checked.append("courtlistener")
            outcome = await _lookup_courtlistener(citation, client)
            if outcome.found:
                result.matches["courtlistener"] = outcome.payload or {}
                result.status = VerificationStatus.REAL
                result.confidence = 1.0
                resolved = True
            elif outcome.definitive:
                negative_definitive_count += 1

        # ---- DOI path: Crossref → Semantic Scholar → OpenAlex ----
        if citation.doi:
            await _try_doi_source("crossref", _lookup_crossref_doi)
            await _try_doi_source("semantic_scholar", _lookup_s2_doi)
            await _try_doi_source("openalex", _lookup_openalex_doi)

        # ---- Title fallback: Crossref → S2 → OpenAlex ----
        # Step 0 hand-check (§11 amendment #4): legal citations must NOT fall
        # through to academic title-search. Those indexes don't cover US case
        # law, so their "definitive no" votes were inflating FABRICATED on
        # real cases CourtListener can't reach (district-court docket orders
        # outside CL's indexed-opinion set, e.g. Mata v. Avianca).
        if not resolved and citation.title and domain != "legal":
            await _try_title_source("crossref_title", _lookup_crossref_title)
            await _try_title_source("s2_title", _lookup_s2_title)
            await _try_title_source("openalex_title", _lookup_openalex_title)

        # FABRICATED requires at least one definitive negative AND no positive match
        if not resolved and negative_definitive_count >= 1 and sources_attempted > 0:
            result.status = VerificationStatus.FABRICATED
            result.confidence = 0.6 + 0.2 * min(negative_definitive_count, 2)

        if cache and cache_key and result.status != VerificationStatus.UNVERIFIABLE:
            # Only cache definitive results — don't cache transient network failures
            cache.set(cache_key, result.to_cache_dict())
        return result
    finally:
        if own_client:
            await client.aclose()
