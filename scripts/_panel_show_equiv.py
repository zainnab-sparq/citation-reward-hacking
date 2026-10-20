"""Scratch: print the new gap_summary + search_count diagnostic (2026-06-11)."""
import json
from pathlib import Path

r = json.loads(Path("data/analysis/phase_b/batch_equivalence.json").read_text(encoding="utf-8"))
print("gap_summary:", json.dumps(r["gap_summary"], indent=1))
sc = r["search_count"]
print("search_count overall:", json.dumps(sc["overall"], indent=1))
print("skipped (no count):", sc["n_skipped_no_count"])

powered = [s for s in sc["per_stratum"] if min(s["n_subprocess"], s["n_workflow"]) >= 30]
powered.sort(key=lambda s: abs(s["delta_mean"]), reverse=True)
print("\nwell-powered strata by |search-count delta| (wf - sub):")
for s in powered:
    print(f"  {s['condition']:>20} {s['cell']:>12} {s['domain']:>8} "
          f"n={s['n_subprocess']}/{s['n_workflow']} "
          f"mean={s['mean_subprocess']:.2f}/{s['mean_workflow']:.2f} "
          f"d={s['delta_mean']:+.2f}")

# The two verdict-discordant strata called out by the panel:
focus = [("citations_expected", "known_real", "legal"),
         ("citations_expected", "obscure_real", "medical")]
print("\npanel focus strata:")
for s in sc["per_stratum"]:
    if (s["condition"], s["cell"], s["domain"]) in focus:
        print(f"  {s['condition']} x {s['cell']} x {s['domain']}: "
              f"mean_sub={s['mean_subprocess']:.2f} mean_wf={s['mean_workflow']:.2f} "
              f"delta={s['delta_mean']:+.2f}")
