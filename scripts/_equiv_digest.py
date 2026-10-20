import json

rep = json.load(open("data/analysis/phase_b/batch_equivalence.json", encoding="utf-8"))

oc = rep["overall_chi_square"]
print("=== OVERALL (pooled, composition-confounded) ===")
for arm in ("subprocess", "workflow"):
    t = oc["table"][arm]
    n = sum(t.values())
    props = {k: round(v / n, 3) for k, v in t.items()}
    print(f"  {arm:11s} n={n:5d}  {props}")
print(f"  chi2={oc['chi2']:.2f} p={oc['p']:.3g}")

print("\n=== PER-STRATUM (confound-controlled), sorted by max_abs_delta ===")
strata = [s for s in rep["per_stratum"] if not s["insufficient"]]
strata.sort(key=lambda s: s["max_abs_delta"], reverse=True)
print(f"{'cond':20s} {'cell':13s} {'dom':9s} {'nS':>4s} {'nW':>4s} {'maxd':>6s}  argmax-status")
for s in strata[:12]:
    d = s["abs_deltas"]
    arg = max(d, key=d.get)
    print(f"{s['condition']:20s} {s['cell']:13s} {s['domain']:9s} "
          f"{s['n_subprocess']:4d} {s['n_workflow']:4d} {s['max_abs_delta']:6.3f}  {arg}")

# weighted mean abs delta (by total stratum n) to summarize magnitude
tot = sum(s["n_subprocess"] + s["n_workflow"] for s in strata)
wmean = sum(s["max_abs_delta"] * (s["n_subprocess"] + s["n_workflow"]) for s in strata) / tot
print(f"\nn comparable strata: {len(strata)} | insufficient: {rep['n_strata']-len(strata)}")
print(f"n-weighted mean of per-stratum max_abs_delta: {wmean:.4f}")
print(f"unweighted mean: {sum(s['max_abs_delta'] for s in strata)/len(strata):.4f}")

# how many strata have small n in either arm (noise-prone)?
small = [s for s in strata if min(s["n_subprocess"], s["n_workflow"]) < 30]
print(f"strata with min-arm-n < 30 (noise-prone): {len(small)}")
big = [s for s in strata if min(s["n_subprocess"], s["n_workflow"]) >= 30]
if big:
    bmean = sum(s["max_abs_delta"] for s in big) / len(big)
    bmax = max(s["max_abs_delta"] for s in big)
    print(f"among {len(big)} well-powered strata (min-arm-n>=30): mean maxd={bmean:.4f}, max maxd={bmax:.4f}")
