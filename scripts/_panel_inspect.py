"""Scratch: schema sanity for the panel zero-cost diagnostics (2026-06-11)."""
import json
from collections import Counter
from pathlib import Path

from src.analysis.batch_equivalence import _read_jsonl, partition_provenance

JUDG = Path("data/judgments/phase_b/claude-subagent-sonnet.jsonl")
BAK = Path("data/judgments/phase_b/claude-subagent-sonnet.jsonl.before_repurge_20260608_041032.bak")
EXTR = Path("data/extractions/phase_b/extractions.jsonl")

judgments = _read_jsonl(JUDG)
prov = partition_provenance(_read_jsonl(BAK))

print("judgment keys:", sorted(judgments[0].keys()))
sc_missing = Counter()
sc_present = Counter()
sc_nonint = 0
for j in judgments:
    p = prov.get(j["custom_id"], "UNATTRIBUTED")
    sc = j.get("judge_search_count")
    if isinstance(sc, int):
        sc_present[p] += 1
    else:
        sc_missing[p] += 1
        sc_nonint += 1
print("search_count present by arm:", dict(sc_present))
print("search_count missing/non-int by arm:", dict(sc_missing))

ext = _read_jsonl(EXTR)
print("extraction keys:", sorted(ext[0].keys()))
row0 = ext[0]
cits = row0.get("citations") or []
if cits:
    print("citation keys:", sorted(cits[0].keys()))
n_titleless = 0
titleless_examples = []
for r in ext:
    for c in r.get("citations") or []:
        if c.get("title") is None and c.get("case_name"):
            n_titleless += 1
            if len(titleless_examples) < 3:
                titleless_examples.append(
                    {k: c.get(k) for k in ("title", "case_name", "reporter", "year")}
                )
print("title-less legal citations:", n_titleless)
for e in titleless_examples:
    print("  ", e)
