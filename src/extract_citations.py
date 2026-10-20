"""LLM-assisted citation extraction.

Replaces the brittle regex extractor in src/pilot.py with a Sonnet-based
structured-output extractor. The Phase A pilot showed that regex misses
citations embedded mid-sentence, with quoted titles, or in non-APA formats —
all of which a capable LLM handles reliably via tool-use structured output.

Production uses AnthropicExtractorClient (forces a tool call so the response
is always valid JSON); tests inject FakeExtractorClient with canned outputs.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from src.verify import Citation, Domain

# Models often write arXiv references as "arXiv:1810.04805" or "arXiv (1810.04805)"
# rather than spelling out the full DOI. Without recognizing the bare ID, the
# verifier falls back to flaky title search and false-fabricates real arXiv papers.
_ARXIV_ID_RE = re.compile(r"\barXiv[\s:(\[]+(\d{4}\.\d{4,5})\b", re.IGNORECASE)

# Default Sonnet snapshot for production extraction. Override per-call when
# pinning for paper-reported runs (see plan §7.9 / §14).
DEFAULT_EXTRACTOR_MODEL = "claude-sonnet-4-6"

_EXTRACTION_TOOL = {
    "name": "record_citations",
    "description": (
        "Record every citation found in the response text. Include cases where "
        "the citation is embedded mid-sentence, has a quoted title, uses "
        "non-standard formatting, or is a legal case citation. If no citations "
        "are present, return an empty list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "citations": {
                "type": "array",
                "description": "All citations found in the text.",
                "items": {
                    "type": "object",
                    "properties": {
                        "raw_span": {
                            "type": "string",
                            "description": "The exact text span containing this citation.",
                        },
                        "authors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Author surnames in citation order. Empty for legal cases.",
                        },
                        "year": {
                            "type": ["integer", "null"],
                            "description": "Publication or decision year.",
                        },
                        "title": {
                            "type": "string",
                            "description": "Title of the work or case name.",
                        },
                        "venue": {
                            "type": ["string", "null"],
                            "description": "Journal, conference, publisher, or court reporter name.",
                        },
                        "doi": {
                            "type": ["string", "null"],
                            "description": "DOI without the https://doi.org/ prefix, if present.",
                        },
                        "case_name": {
                            "type": ["string", "null"],
                            "description": "Legal case name (e.g., 'Miranda v. Arizona'). Null for non-legal.",
                        },
                        "reporter": {
                            "type": ["string", "null"],
                            "description": "Reporter citation (e.g., '384 U.S. 436'). Null for non-legal.",
                        },
                    },
                    "required": ["raw_span", "title"],
                },
            }
        },
        "required": ["citations"],
    },
}

_EXTRACTION_PROMPT = (
    "Extract every citation from the following response text. A citation is any "
    "reference to a published work, paper, book, court case, clinical trial, or "
    "similar (whether formatted as APA, IEEE, Bluebook, embedded mid-sentence, "
    "or with a quoted title) that the response ASSERTS or claims as a source.\n\n"
    "IMPORTANT — abstention guard: Do not extract a citation when it appears "
    "only inside refusal, abstention, or uncertainty language. Common patterns "
    "to skip:\n"
    "  - 'I cannot verify <citation>'\n"
    "  - 'I am not aware of <citation>'\n"
    "  - 'I don't have / I do not have reliable information about <citation>'\n"
    "  - 'no such case / paper / trial as <citation> exists'\n"
    "  - 'I will not fabricate <citation>'\n"
    "  - 'I cannot provide the citation you requested'\n"
    "  - sentences in which the response is DENYING the citation rather than "
    "claiming it as a source\n"
    "If the response refuses to cite a work but names it only to identify what "
    "is being refused, that is NOT a citation — do not extract it. If the "
    "response instead asserts a real or claimed source (even tentatively, e.g. "
    "'see also X v. Y'), that IS a citation — extract it.\n\n"
    "Domain hint: {domain}.\n\n"
    "Response text:\n---\n{text}\n---\n\n"
    "Use the record_citations tool to return your structured output. "
    "If no citations are present (including responses that consist entirely of "
    "abstention or refusal), call the tool with an empty list."
)


@dataclass(frozen=True)
class ExtractedCitation:
    raw_span: str
    title: str | None = None
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    case_name: str | None = None
    reporter: str | None = None

    def to_verify_citation(self, domain: Domain) -> Citation:
        """Convert to the verify.Citation type used by the verification pipeline."""
        return Citation(
            raw_text=self.raw_span,
            domain=domain,
            title=self.title,
            authors=self.authors,
            year=self.year,
            venue=self.venue,
            doi=self.doi,
            case_name=self.case_name,
            reporter=self.reporter,
        )

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExtractedCitation:
        doi = d.get("doi")
        if not doi:
            arxiv_match = _ARXIV_ID_RE.search(d.get("raw_span", ""))
            if arxiv_match:
                doi = f"10.48550/arXiv.{arxiv_match.group(1)}"
        return cls(
            raw_span=d["raw_span"],
            title=d.get("title"),
            authors=tuple(d.get("authors") or ()),
            year=d.get("year"),
            venue=d.get("venue"),
            doi=doi,
            case_name=d.get("case_name"),
            reporter=d.get("reporter"),
        )


class CitationExtractorClient(Protocol):
    """Pluggable backend that returns the LLM's structured citation list.

    Production: AnthropicExtractorClient calls Sonnet with forced tool use.
    Tests: FakeExtractorClient returns canned dicts so the suite stays hermetic.
    """

    async def extract(self, text: str, domain: Domain) -> list[dict[str, Any]]:
        """Return a list of citation dicts matching the record_citations schema."""
        ...


class FakeExtractorClient:
    """Returns canned citation dicts (or raises) for testing."""

    def __init__(
        self,
        canned: list[dict[str, Any]] | None,
        raise_on_call: Exception | None = None,
    ):
        self._canned = canned
        self._raise = raise_on_call

    async def extract(self, text: str, domain: Domain) -> list[dict[str, Any]]:
        if self._raise is not None:
            raise self._raise
        return list(self._canned or [])


class AnthropicExtractorClient:
    """Production extractor — Sonnet 4.6 with forced tool use for valid JSON output."""

    def __init__(
        self,
        client: Any | None = None,
        model_id: str = DEFAULT_EXTRACTOR_MODEL,
        max_output_tokens: int = 2048,
    ):
        if client is None:
            from anthropic import AsyncAnthropic
            client = AsyncAnthropic()
        self._client = client
        self._model_id = model_id
        self._max_tokens = max_output_tokens

    async def extract(self, text: str, domain: Domain) -> list[dict[str, Any]]:
        prompt = _EXTRACTION_PROMPT.format(domain=domain, text=text)
        response = await self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            tools=[_EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "record_citations"},
            messages=[{"role": "user", "content": prompt}],
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == "record_citations":
                payload = block.input
                if isinstance(payload, str):
                    payload = json.loads(payload)
                return list(payload.get("citations", []))
        return []


async def extract_citations_llm(
    text: str,
    domain: Domain,
    client: CitationExtractorClient,
) -> list[ExtractedCitation]:
    """Extract citations from a model's response using the LLM extractor.

    Returns [] on extractor errors rather than raising — a malformed extraction
    must not crash the verification pipeline mid-batch.
    """
    try:
        raw = await client.extract(text, domain=domain)
    except Exception:
        return []
    extracted: list[ExtractedCitation] = []
    for item in raw:
        try:
            extracted.append(ExtractedCitation.from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return extracted
