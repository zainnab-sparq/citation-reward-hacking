"""HARD legal-unanswerable cross-check gate (pre-reg §8).

The 100 legal *unanswerable* items are researcher-constructed fictitious cases
with facially-impossible reporter citations (e.g. "999 U.S. 22187"). OpenAlex,
the SPECTER arm of the negative-existence proof, does not index US case law, so
that proof gives near-zero protection for the legal cell. This module is the
gate that actually certifies non-existence: it runs every legal unanswerable's
reporter citation through CourtListener's Citation Lookup API and confirms none
resolves to a real opinion.

This is the INVERSE of normal citation verification (`src/verify.py`): there a
match is the good outcome; here a match means a constructed "unanswerable" is
secretly a real case and must be removed/replaced before any legal fabrication
rate is reported.

Endpoint contract (CourtListener API v4 Citation Lookup):
  POST https://www.courtlistener.com/api/rest/v4/citation-lookup/
  Auth: ``Authorization: Token <token>``
  Body: ``text`` (up to 64,000 chars, up to 250 citations) OR volume/reporter/page
  Throttle: 60 *valid* citations / minute; 429 carries a wait hint
  Returns a JSON array; one object per parsed citation with a per-citation
  ``status`` mirroring HTTP codes (200/300 = matched a real cluster, 404 = valid
  but absent, 400 = reporter not in system) and a ``clusters`` list.

Run: ``python -m src.dataset_construction.legal_crosscheck`` (needs a token).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import httpx

CITATION_LOOKUP_URL = "https://www.courtlistener.com/api/rest/v4/citation-lookup/"
_TOKEN_ENV = "COURTLISTENER_API_TOKEN"

# CourtListener throttles to 60 *valid* citations per minute and accepts up to
# 250 per request. We send batches under the per-minute cap and pause one full
# window between batches so a 100-item gate run never trips the 429 throttle.
_MAX_CITATIONS_PER_BATCH = 50
_THROTTLE_SECONDS = 60.0
_DEFAULT_TIMEOUT = httpx.Timeout(30.0)
_MAX_RETRIES = 3

# Per-citation status codes (mirror HTTP). A real cluster is the ground truth of
# "this case exists"; status is only the tie-breaker when clusters is empty.
_NO_MATCH_STATUSES = frozenset({400, 404})

_DEFAULT_DATASET = "data/questions/frozen_900q_legal_amended.jsonl"
_DEFAULT_OUTPUT = "data/verifications/legal_unanswerable_crosscheck.jsonl"

SleepFn = Callable[[float], Awaitable[None]]


class CourtListenerTokenMissing(RuntimeError):
    """Raised when COURTLISTENER_API_TOKEN is not set.

    The gate must certify non-existence positively; running anonymously would
    silently degrade to a non-definitive result and could let a real case slip
    through, so we refuse to run without a token rather than guess.
    """


class Verdict(StrEnum):
    REAL_MATCH = "real_match"      # a real opinion exists -> the item is NOT unanswerable
    NO_MATCH = "no_match"          # confirmed absent from CourtListener
    UNVERIFIABLE = "unverifiable"  # throttled / unparsed / contradictory -> must retry


class GateState(StrEnum):
    PASS = "pass"              # every item confirmed non-existent
    FAIL = "fail"             # at least one item resolved to a real case
    INCOMPLETE = "incomplete"  # no real matches, but some items unverifiable


@dataclass(frozen=True)
class LegalUnanswerable:
    id: str
    reporter: str
    case_name: str


@dataclass
class CrosscheckRecord:
    id: str
    reporter: str
    case_name: str
    status: int
    verdict: Verdict
    cluster_ids: list[int]
    normalized_citations: list[str]


@dataclass
class GateResult:
    state: GateState
    n_total: int
    n_no_match: int
    n_real_match: int
    n_unverifiable: int
    real_match_ids: list[str]
    unverifiable_ids: list[str]


def load_legal_unanswerables(path: str | Path) -> list[LegalUnanswerable]:
    """Load the legal-unanswerable items from a frozen-dataset JSONL file."""
    items: list[LegalUnanswerable] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("domain") != "legal" or row.get("cell") != "unanswerable":
            continue
        citation = row.get("expected_citation") or {}
        items.append(
            LegalUnanswerable(
                id=row["id"],
                reporter=citation.get("reporter") or "",
                case_name=citation.get("case_name") or citation.get("title") or "",
            )
        )
    return items


def _normalize_cite(value: str | None) -> str:
    """Collapse internal whitespace so "12  F.4th  8847" == "12 F.4th 8847"."""
    return " ".join((value or "").split())


def build_lookup_payload(reporters: list[str]) -> dict[str, str]:
    """One reporter citation per line; CourtListener parses each independently."""
    return {"text": "\n".join(reporters)}


def _cluster_ids(result: dict[str, Any]) -> list[int]:
    return [c["id"] for c in (result.get("clusters") or []) if c.get("id") is not None]


def classify_result(result: dict[str, Any]) -> Verdict:
    """Map one citation-lookup result object to a verdict.

    A non-empty ``clusters`` is the decisive signal that a real opinion exists.
    Absent clusters, a definitive not-found status (400/404) is NO_MATCH; any
    other state (429 throttle, or a contradictory 200-with-no-clusters) is
    UNVERIFIABLE and must be retried rather than counted as a pass.
    """
    if result.get("clusters"):
        return Verdict.REAL_MATCH
    if result.get("status") in _NO_MATCH_STATUSES:
        return Verdict.NO_MATCH
    return Verdict.UNVERIFIABLE


def align_results(
    records: list[LegalUnanswerable], results: list[dict[str, Any]]
) -> list[CrosscheckRecord]:
    """Join API results back to records by normalized citation string.

    Order is not relied upon: each result is indexed by every citation string it
    exposes (raw + normalized). A record with no matching result (eyecite failed
    to parse its reporter) is UNVERIFIABLE, never silently dropped.
    """
    index: dict[str, dict[str, Any]] = {}
    for result in results:
        keys = set()
        if result.get("citation"):
            keys.add(_normalize_cite(result["citation"]))
        for normalized in result.get("normalized_citations") or []:
            keys.add(_normalize_cite(normalized))
        for key in keys:
            index.setdefault(key, result)

    aligned: list[CrosscheckRecord] = []
    for record in records:
        result = index.get(_normalize_cite(record.reporter))
        if result is None:
            aligned.append(
                CrosscheckRecord(
                    id=record.id,
                    reporter=record.reporter,
                    case_name=record.case_name,
                    status=0,
                    verdict=Verdict.UNVERIFIABLE,
                    cluster_ids=[],
                    normalized_citations=[],
                )
            )
            continue
        aligned.append(
            CrosscheckRecord(
                id=record.id,
                reporter=record.reporter,
                case_name=record.case_name,
                status=int(result.get("status") or 0),
                verdict=classify_result(result),
                cluster_ids=_cluster_ids(result),
                normalized_citations=[
                    _normalize_cite(c) for c in (result.get("normalized_citations") or [])
                ],
            )
        )
    return aligned


def evaluate_gate(records: list[CrosscheckRecord]) -> GateResult:
    """PASS only if every item is NO_MATCH; any real match FAILs the gate."""
    real_match_ids = [r.id for r in records if r.verdict == Verdict.REAL_MATCH]
    unverifiable_ids = [r.id for r in records if r.verdict == Verdict.UNVERIFIABLE]
    n_no_match = sum(1 for r in records if r.verdict == Verdict.NO_MATCH)

    if real_match_ids:
        state = GateState.FAIL
    elif unverifiable_ids:
        state = GateState.INCOMPLETE
    else:
        state = GateState.PASS

    return GateResult(
        state=state,
        n_total=len(records),
        n_no_match=n_no_match,
        n_real_match=len(real_match_ids),
        n_unverifiable=len(unverifiable_ids),
        real_match_ids=real_match_ids,
        unverifiable_ids=unverifiable_ids,
    )


def _retry_after_seconds(response: httpx.Response) -> float:
    header = response.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return _THROTTLE_SECONDS


async def _post_batch(
    client: httpx.AsyncClient,
    payload: dict[str, str],
    headers: dict[str, str],
    sleep: SleepFn,
) -> list[dict[str, Any]]:
    """POST one batch, honoring a 429 throttle with bounded retries."""
    for _ in range(_MAX_RETRIES + 1):
        response = await client.post(
            CITATION_LOOKUP_URL, data=payload, headers=headers, timeout=_DEFAULT_TIMEOUT
        )
        if response.status_code == 429:
            await sleep(_retry_after_seconds(response))
            continue
        response.raise_for_status()
        return response.json()
    raise httpx.HTTPError("CourtListener citation-lookup throttled after retries")


async def run_crosscheck(
    records: list[LegalUnanswerable],
    client: httpx.AsyncClient,
    *,
    sleep: SleepFn = asyncio.sleep,
    batch_size: int = _MAX_CITATIONS_PER_BATCH,
) -> list[CrosscheckRecord]:
    """Cross-check every record against CourtListener Citation Lookup.

    Requires COURTLISTENER_API_TOKEN. Batches under the per-minute throttle and
    pauses one window between batches.
    """
    token = os.environ.get(_TOKEN_ENV)
    if not token:
        raise CourtListenerTokenMissing(
            f"Set {_TOKEN_ENV} to run the legal-unanswerable cross-check gate "
            "(get a free token at courtlistener.com)."
        )
    headers = {"Authorization": f"Token {token}"}

    aligned: list[CrosscheckRecord] = []
    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]
    for batch_index, batch in enumerate(batches):
        if batch_index > 0:
            await sleep(_THROTTLE_SECONDS)
        payload = build_lookup_payload([r.reporter for r in batch])
        results = await _post_batch(client, payload, headers, sleep)
        aligned.extend(align_results(batch, results))
    return aligned


def write_crosscheck_log(
    records: list[CrosscheckRecord], gate: GateResult, path: str | Path
) -> None:
    """Write one JSONL line per record plus a trailing summary line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "id": r.id,
                "reporter": r.reporter,
                "case_name": r.case_name,
                "status": r.status,
                "verdict": r.verdict.value,
                "cluster_ids": r.cluster_ids,
                "normalized_citations": r.normalized_citations,
            }
        )
        for r in records
    ]
    lines.append(
        json.dumps(
            {
                "type": "summary",
                "gate_state": gate.state.value,
                "n_total": gate.n_total,
                "n_no_match": gate.n_no_match,
                "n_real_match": gate.n_real_match,
                "n_unverifiable": gate.n_unverifiable,
                "real_match_ids": gate.real_match_ids,
                "unverifiable_ids": gate.unverifiable_ids,
            }
        )
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=_DEFAULT_DATASET)
    parser.add_argument("--out", default=_DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    records = load_legal_unanswerables(args.dataset)

    async def _run() -> list[CrosscheckRecord]:
        async with httpx.AsyncClient() as client:
            return await run_crosscheck(records, client)

    crosschecked = asyncio.run(_run())
    gate = evaluate_gate(crosschecked)
    write_crosscheck_log(crosschecked, gate, args.out)

    print(
        f"Legal-unanswerable cross-check: {gate.state.value.upper()} "
        f"({gate.n_no_match}/{gate.n_total} confirmed non-existent, "
        f"{gate.n_real_match} real matches, {gate.n_unverifiable} unverifiable)"
    )
    if gate.real_match_ids:
        print("REAL MATCHES (must remove/replace):", ", ".join(gate.real_match_ids))
    if gate.unverifiable_ids:
        print("UNVERIFIABLE (must retry):", ", ".join(gate.unverifiable_ids))
    return 0 if gate.state == GateState.PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
