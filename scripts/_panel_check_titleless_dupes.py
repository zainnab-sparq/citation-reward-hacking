"""Scratch: did any of the 34 title-less legal citations escape dedup?

dedup_citations never merges empty-title rows (by design), so a title-less
legal citation could double-count against another citation of the SAME case in
the SAME generation. Checks case_name (normalized) and reporter matches within
each affected extraction record.
"""
import re
from pathlib import Path

from src.analysis.aggregate import _read_jsonl

EXTR = Path("data/extractions/phase_b/extractions.jsonl")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def norm(s):
    if not isinstance(s, str):
        return ""
    return _WS.sub(" ", _PUNCT.sub(" ", s.lower())).strip()


n_titleless = 0
n_dupe_risk = 0
for rec in _read_jsonl(EXTR):
    cits = rec.get("citations") or []
    titleless = [c for c in cits if c.get("title") is None and c.get("case_name")]
    if not titleless:
        continue
    n_titleless += len(titleless)
    for t in titleless:
        for other in cits:
            if other is t:
                continue
            same_case = norm(t.get("case_name")) and (
                norm(t.get("case_name")) == norm(other.get("case_name"))
                or norm(t.get("case_name")) == norm(other.get("title"))
            )
            same_reporter = t.get("reporter") and t.get("reporter") == other.get("reporter")
            if same_case or same_reporter:
                n_dupe_risk += 1
                print(f"DUPE RISK in {rec['question_id']}/{rec['condition']}/t{rec['trial_id']}:")
                print(f"  titleless: case_name={t.get('case_name')!r} reporter={t.get('reporter')!r}")
                print(f"  other:     title={other.get('title')!r} "
                      f"case_name={other.get('case_name')!r} reporter={other.get('reporter')!r}")

print(f"\ntitle-less legal citations: {n_titleless}")
print(f"within-generation duplicate risks: {n_dupe_risk}")
