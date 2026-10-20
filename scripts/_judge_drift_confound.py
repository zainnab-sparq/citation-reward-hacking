"""Is the search-grounding degradation confounded with the experimental factors?

custom_id: <domain>_<qtype>_<NNNN>__<model>__<condition>__t0__c<n>
qtype in {known_real, obscure_real, unanswerable} is the stratifier that drives
fabrication rates, so it must be composition-matched before any era comparison means
anything. If EARLY and LATE match on all factors, a verdict-rate shift is an
instrument shift, not a content shift.
"""
import collections
import json

PATH = "data/judgments/phase_c/claude-subagent-sonnet.jsonl"
SC_KEY = "judge_search_count"
STATUS_KEY = "judge_status"
SPLIT = 33 * 500  # regime change observed at block 33


def parse(cid):
    parts = cid.split("__")
    qid, model, cond = parts[0], parts[1], parts[2]
    body = qid.rsplit("_", 1)[0]          # drop trailing _NNNN
    domain, qtype = body.split("_", 1)     # legal / unanswerable
    return domain, qtype, model, cond


rows = [json.loads(line) for line in open(PATH, encoding="utf-8")]
early, late = rows[:SPLIT], rows[SPLIT:]

FACTORS = ("domain", "qtype", "model", "condition")


def profile(chunk):
    ctrs = {f: collections.Counter() for f in FACTORS}
    for r in chunk:
        for f, v in zip(FACTORS, parse(r["custom_id"])):
            ctrs[f][v] += 1
    return ctrs


pe, pl = profile(early), profile(late)
print(f"composition EARLY (n={len(early)}) vs LATE (n={len(late)}), pp = late - early\n")
for f in FACTORS:
    keys = sorted(set(pe[f]) | set(pl[f]), key=lambda k: -pe[f].get(k, 0))
    print(f"  [{f}]")
    for k in keys:
        a = 100 * pe[f].get(k, 0) / len(early)
        b = 100 * pl[f].get(k, 0) / len(late)
        print(f"    {k:<38} {a:>5.1f}% -> {b:>5.1f}%  ({b - a:+.1f}pp)")

print("\nverdict rates by era:")
for label, chunk in (("EARLY", early), ("LATE", late)):
    st = collections.Counter(r.get(STATUS_KEY) for r in chunk)
    n = len(chunk)
    scs = [r.get(SC_KEY) for r in chunk if isinstance(r.get(SC_KEY), int)]
    print(f"  {label:<6} " + ", ".join(f"{k}={100 * v / n:.1f}%" for k, v in sorted(st.items()))
          + f", mean_sc={sum(scs) / len(scs):.2f}")

print("\nverdict rates WITHIN qtype (controls for the strongest content driver):")
print(f"  {'qtype':<14} {'era':<6} {'n':>6} {'REAL%':>7} {'FAB%':>7} {'UNV%':>7} {'mean_sc':>8}")
for qtype in ("known_real", "obscure_real", "unanswerable"):
    for label, chunk in (("EARLY", early), ("LATE", late)):
        sub = [r for r in chunk if parse(r["custom_id"])[1] == qtype]
        if not sub:
            continue
        st = collections.Counter(r.get(STATUS_KEY) for r in sub)
        scs = [r.get(SC_KEY) for r in sub if isinstance(r.get(SC_KEY), int)]
        n = len(sub)
        print(f"  {qtype:<14} {label:<6} {n:>6} {100 * st['REAL'] / n:>6.1f}% "
              f"{100 * st['FABRICATED'] / n:>6.1f}% {100 * st['UNVERIFIABLE'] / n:>6.1f}% "
              f"{sum(scs) / len(scs):>8.2f}")
