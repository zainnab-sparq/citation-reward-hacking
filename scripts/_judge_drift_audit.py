"""Integrity check + search_count audit for the Phase C J2a judgment checkpoint.

Rows are in append order, so index order == batch order == chronological order.
That lets us test whether evidence-grounding degrades over the life of a session.
"""
import collections
import json

PATH = "data/judgments/phase_c/claude-subagent-sonnet.jsonl"
SC_KEY = "judge_search_count"
STATUS_KEY = "judge_status"
BLOCK = 500

rows = [json.loads(line) for line in open(PATH, encoding="utf-8")]
ids = [r["custom_id"] for r in rows]
dupes = sum(1 for _, v in collections.Counter(ids).items() if v > 1)
print("rows", len(ids), "unique", len(set(ids)), "dupes", dupes)

counts = collections.Counter(r.get(SC_KEY) for r in rows)
print("search_count distribution:", dict(sorted(counts.items(), key=lambda kv: (kv[0] is None, kv[0]))))

zero = [r for r in rows if r.get(SC_KEY) == 0]
print("search_count==0:", len(zero), f"({100 * len(zero) / len(rows):.2f}%)")
print("  status split (search_count==0):", dict(collections.Counter(r.get(STATUS_KEY) for r in zero)))
print("  status split (ALL rows):", dict(collections.Counter(r.get(STATUS_KEY) for r in rows)))

print("\nper-block trend (block = %d rows, append order ~= chronological):" % BLOCK)
print(f"{'block':>6} {'rows':>6} {'sc0%':>7} {'mean_sc':>8} {'REAL%':>7}")
for start in range(0, len(rows), BLOCK):
    chunk = rows[start:start + BLOCK]
    scs = [r.get(SC_KEY) for r in chunk if isinstance(r.get(SC_KEY), int)]
    n_zero = sum(1 for s in scs if s == 0)
    n_real = sum(1 for r in chunk if r.get(STATUS_KEY) == "REAL")
    mean_sc = sum(scs) / len(scs) if scs else float("nan")
    print(f"{start // BLOCK:>6} {len(chunk):>6} {100 * n_zero / len(chunk):>6.1f}% "
          f"{mean_sc:>8.2f} {100 * n_real / len(chunk):>6.1f}%")

TAIL = 2000
tail = rows[-TAIL:]
tail_scs = [r.get(SC_KEY) for r in tail if isinstance(r.get(SC_KEY), int)]
head = rows[:TAIL]
head_scs = [r.get(SC_KEY) for r in head if isinstance(r.get(SC_KEY), int)]
print(f"\nfirst {TAIL}: sc0={100 * sum(1 for s in head_scs if s == 0) / len(head_scs):.1f}% "
      f"mean_sc={sum(head_scs) / len(head_scs):.2f}")
print(f"last  {TAIL}: sc0={100 * sum(1 for s in tail_scs if s == 0) / len(tail_scs):.1f}% "
      f"mean_sc={sum(tail_scs) / len(tail_scs):.2f}")
