"""Three-era comparison after the 2026-08-05 grounding recovery.

Grounding recovered at block 48 (mean_sc back to ~2.09) but REAL did NOT fully
revert to the early-era rate. Two competing explanations:
  (1) a SECOND instrument change independent of search grounding, or
  (2) composition — the tail of the queue holds different items.
This separates them by profiling composition per era and, critically, by
comparing verdict rates WITHIN qtype (the dominant content driver).

Eras (append order == chronological):
  EARLY     rows      0..16,499  (blocks  0-32)  well-grounded
  DEGRADED  rows 16,500..23,999  (blocks 33-47)  sc0 11-15%, mean_sc 1.61-1.71
  RECOVERED rows 24,000..        (blocks 48+)    sc0 0.4-1.0%, mean_sc ~2.09
"""
import collections
import json

PATH = "data/judgments/phase_c/claude-subagent-sonnet.jsonl"
SC_KEY = "judge_search_count"
STATUS_KEY = "judge_status"

DEGRADED_START = 33 * 500   # 16,500
RECOVERED_START = 48 * 500  # 24,000

FACTORS = ("domain", "qtype", "model", "condition")


def parse(cid):
    parts = cid.split("__")
    qid, model, cond = parts[0], parts[1], parts[2]
    body = qid.rsplit("_", 1)[0]
    domain, qtype = body.split("_", 1)
    return domain, qtype, model, cond


def rates(chunk):
    n = len(chunk)
    st = collections.Counter(r.get(STATUS_KEY) for r in chunk)
    scs = [r.get(SC_KEY) for r in chunk if isinstance(r.get(SC_KEY), int)]
    return {
        "n": n,
        "REAL": 100 * st["REAL"] / n,
        "FAB": 100 * st["FABRICATED"] / n,
        "UNV": 100 * st["UNVERIFIABLE"] / n,
        "mean_sc": sum(scs) / len(scs) if scs else float("nan"),
    }


rows = [json.loads(line) for line in open(PATH, encoding="utf-8")]
eras = {
    "EARLY": rows[:DEGRADED_START],
    "DEGRADED": rows[DEGRADED_START:RECOVERED_START],
    "RECOVERED": rows[RECOVERED_START:],
}

print("=== overall verdict rates by era ===")
print(f"  {'era':<10} {'n':>6} {'REAL%':>7} {'FAB%':>7} {'UNV%':>7} {'mean_sc':>8}")
for name, chunk in eras.items():
    r = rates(chunk)
    print(f"  {name:<10} {r['n']:>6} {r['REAL']:>6.1f}% {r['FAB']:>6.1f}% "
          f"{r['UNV']:>6.1f}% {r['mean_sc']:>8.2f}")

print("\n=== composition (pp), to test the 'different items' explanation ===")
profiles = {}
for name, chunk in eras.items():
    ctrs = {f: collections.Counter() for f in FACTORS}
    for r in chunk:
        for f, v in zip(FACTORS, parse(r["custom_id"])):
            ctrs[f][v] += 1
    profiles[name] = ctrs

for f in FACTORS:
    keys = sorted(set().union(*(profiles[e][f] for e in eras)),
                  key=lambda k: -profiles["EARLY"][f].get(k, 0))
    print(f"\n  [{f}]")
    print(f"    {'level':<38} {'EARLY':>7} {'DEGR':>7} {'RECOV':>7} {'R-E':>7}")
    for k in keys:
        pct = {e: 100 * profiles[e][f].get(k, 0) / len(eras[e]) for e in eras}
        print(f"    {k:<38} {pct['EARLY']:>6.1f}% {pct['DEGRADED']:>6.1f}% "
              f"{pct['RECOVERED']:>6.1f}% {pct['RECOVERED'] - pct['EARLY']:>+6.1f}")

print("\n=== verdict rates WITHIN qtype (composition-controlled) ===")
print("If RECOVERED still sits below EARLY inside every qtype, composition does")
print("NOT explain the residual gap and a second instrument change is implicated.")
print(f"\n  {'qtype':<14} {'era':<10} {'n':>6} {'REAL%':>7} {'FAB%':>7} {'UNV%':>7} {'mean_sc':>8}")
for qtype in ("known_real", "obscure_real", "unanswerable"):
    for name, chunk in eras.items():
        sub = [r for r in chunk if parse(r["custom_id"])[1] == qtype]
        if not sub:
            continue
        r = rates(sub)
        print(f"  {qtype:<14} {name:<10} {r['n']:>6} {r['REAL']:>6.1f}% {r['FAB']:>6.1f}% "
              f"{r['UNV']:>6.1f}% {r['mean_sc']:>8.2f}")
