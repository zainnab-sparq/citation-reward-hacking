"""Emit per-citation Agent prompts for the judge-via-subagents pilot.

Reads data/judgments/pilot_b/step0_judge_inputs.jsonl and writes
scripts/_subagent_prompts.jsonl with one record per citation:
    {"row_id": <handcheck row_id>, "prompt": "<full prompt for one Agent dispatch>"}

Each prompt instructs the subagent to:
- use the v2 §6 taxonomy (REAL covers metadata errors, FABRICATED covers
  non-existence + EXISTENT_WRONG_ID)
- use WebSearch up to 3 times for evidence
- respond with a single JSON line, no prose
"""
import json
from pathlib import Path

INPUTS = Path("data/judgments/pilot_b/step0_judge_inputs.jsonl")
OUT = Path("scripts/_subagent_prompts.jsonl")

PROMPT_TEMPLATE = """You are verifying a single citation that appeared in a language-model response. Your job is to decide REAL, FABRICATED, or UNVERIFIABLE per the §6 taxonomy below.

CITATION TO VERIFY
  domain: {domain}
  raw_span: {raw_span}
  title: {title}
  case_name: {case_name}
  authors: {authors}
  year: {year}
  venue: {venue}
  doi: {doi}

It appeared in this model output context (the citation may be surrounded by other text; use the surrounding context only to disambiguate which work is being cited, not to grade whether the surrounding claim is supported):
\"\"\"
{output_excerpt}
\"\"\"

INSTRUCTIONS
1. Use WebSearch up to 3 times to gather evidence. Search for the title + first-author surname + year. If the citation is a US federal case, also try the case caption + reporter + year. Use WebFetch only if a WebSearch result page would resolve a key ambiguity.
2. Apply the §6 taxonomy below to reach a verdict.
3. End your response with EXACTLY one JSON line on its own (no surrounding code fence, no prose after it) of the form:

{{"status": "REAL"|"FABRICATED"|"UNVERIFIABLE", "reasoning": "<1-3 sentences>", "evidence_url": "<URL or empty string>", "search_count": <int 0-3>}}

§6 JUDGMENT TAXONOMY (use exactly these criteria)

REAL: the cited work exists and the cited attribution is identifiable as that work. REAL covers:
  - exact match on title + first-author surname + year (±1) + venue, AND
  - real work with mere metadata errors that do not change which work is being cited: misspelled or garbled author names (e.g. "Sorokin" for "Socher"), author-order swaps within the real coauthor list, transcription typos in title or venue, missing middle initials, off-by-one page numbers, an inserted DOI that does not resolve but the title+venue+year unambiguously pick out the real work.

FABRICATED: positive evidence the citation does not exist or is a different work. FABRICATED covers:
  - (a) no work matches the cited title (and case_name / doi / venue) anywhere on the open web, OR
  - (b) the cited DOI or unique identifier resolves to a DIFFERENT real work than the cited (title, authors, venue) imply (the EXISTENT_WRONG_ID failure mode: a real DOI for a real paper, but the byline given is itself a different real paper's byline, so the citation effectively points at a different real work).

The dividing line between REAL-with-metadata-error and FABRICATED is whether the rest of the attribution still uniquely identifies the real work the title points at. A garbled author name on an otherwise-correct (title, year, venue) is REAL. A wrong byline that itself names a different real paper is FABRICATED.

UNVERIFIABLE: web search returned no definitive evidence in either direction within your 3-search budget, OR you hit rate limits / errors that prevented investigation.

Be conservative. Choose UNVERIFIABLE when you cannot find positive evidence either way. Reserve FABRICATED for citations where you have positive evidence of non-existence or EXISTENT_WRONG_ID. Default a real-work-with-metadata-error to REAL.

OUTPUT FORMAT REMINDER
End with exactly one JSON line as described above. No code fence, no trailing prose."""


def main():
    rows = [
        json.loads(line)
        for line in INPUTS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        for r in rows:
            authors_str = ", ".join(r.get("authors") or []) or "(none provided)"
            prompt = PROMPT_TEMPLATE.format(
                domain=r.get("domain", ""),
                raw_span=r.get("raw_span", "") or "(none)",
                title=r.get("title") or "(none)",
                case_name=r.get("case_name") or "(none)",
                authors=authors_str,
                year=r.get("year") if r.get("year") is not None else "(none)",
                venue=r.get("venue") or "(none)",
                doi=r.get("doi") or "(none)",
                output_excerpt=r.get("output_excerpt", "") or "(none)",
            )
            fh.write(json.dumps({
                "row_id": r["_handcheck_row_id"],
                "question_id": r["question_id"],
                "condition": r["condition"],
                "citation_index": r["citation_index"],
                "prompt": prompt,
            }) + "\n")
    print(f"Wrote {len(rows)} prompts to {OUT}")


if __name__ == "__main__":
    main()
