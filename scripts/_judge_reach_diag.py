"""Instrument-reach diagnostic for the J2a judge (UNGATED: instrument only).

Answers one question the verdict counts cannot: how often is an UNVERIFIABLE
verdict a statement about the CITATION versus a statement about what the judge
could REACH? Publisher access policy varies by venue and venue varies by domain,
so a domain comparison that reads UNVERIFIABLE as a property of the citation is
confounded by who returned a 403.

The blocked-access count is a KEYWORD PROXY over free-text reasoning, so it is a
LOWER BOUND: it only catches judgments that said so. A judge that hit a wall
silently is invisible here. The paper reports it as a lower bound.

Usage: python scripts/_judge_reach_diag.py <judgments.jsonl>
"""
import json, re, sys
from collections import Counter, defaultdict

BLOCK = re.compile(r"403|forbidden|paywall|cookie wall|blocked|could not be read|failed to load|access denied|not be confirmed because", re.I)

rows = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8') if l.strip()]
by_dom_total, by_dom_unv, by_dom_block = Counter(), Counter(), Counter()
for r in rows:
    dom = r['custom_id'].split('_')[0]
    by_dom_total[dom] += 1
    if r.get('judge_status') == 'UNVERIFIABLE':
        by_dom_unv[dom] += 1
        if BLOCK.search(r.get('judge_reasoning') or ''):
            by_dom_block[dom] += 1

tot_unv = sum(by_dom_unv.values()); tot_blk = sum(by_dom_block.values())
print(f"rows={len(rows)}  UNVERIFIABLE={tot_unv} ({100*tot_unv/len(rows):.2f}%)")
print(f"  of those, reasoning cites blocked/denied access: {tot_blk} ({100*tot_blk/tot_unv:.1f}% of UNV)")
print()
print(f"{'domain':10} {'n':>6} {'UNV':>6} {'UNV%':>7} {'blocked':>8} {'blk% of UNV':>12}")
for d in sorted(by_dom_total):
    n, u, b = by_dom_total[d], by_dom_unv[d], by_dom_block[d]
    print(f"{d:10} {n:6d} {u:6d} {100*u/n:6.2f}% {b:8d} {(f'{100*b/u:.1f}%' if u else 'n/a'):>12}")
