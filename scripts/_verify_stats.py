import json
from collections import Counter

rows = [json.loads(l) for l in open('data/verifications/phase_b/verifications.jsonl', encoding='utf-8')]
n = len(rows)
total_cits = 0
status_counter = Counter()
by_domain = {}
for r in rows:
    qid = r.get('question_id', '')
    dom = qid.split('_')[0] if qid else 'unknown'
    by_domain.setdefault(dom, Counter())
    for s in r.get('citation_statuses', []):
        total_cits += 1
        status_counter[s] += 1
        by_domain[dom][s] += 1

print(f"rows={n}")
print(f"total_citations={total_cits}")
print("\nStatus distribution (overall):")
for s, c in status_counter.most_common():
    print(f"  {s:30s} {c:>6d}  {100*c/total_cits:5.1f}%")

print("\nStatus distribution by domain:")
for dom, ctr in by_domain.items():
    tot = sum(ctr.values())
    if tot == 0:
        continue
    print(f"  {dom} (n={tot}):")
    for s, c in ctr.most_common():
        print(f"    {s:28s} {c:>5d}  {100*c/tot:5.1f}%")

# Sanity: how many will need a judge call under J2a?
j2a_judge_calls = status_counter.get('FABRICATED', 0) + status_counter.get('UNVERIFIABLE', 0)
print(f"\nJ2a-triggering citations (verifier FAB+UNV): {j2a_judge_calls}")
print(f"J2a-skipped (verifier REAL): {status_counter.get('REAL_CONFIRMED', 0) + status_counter.get('REAL', 0)}")
