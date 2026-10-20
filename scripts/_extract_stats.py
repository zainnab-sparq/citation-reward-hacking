import json
rows = [json.loads(l) for l in open('data/extractions/phase_b/extractions.jsonl', encoding='utf-8')]
n = len(rows)
errs = [r for r in rows if r.get('error')]
total_cits = sum(len(r.get('citations', [])) for r in rows)
gens_with_cits = sum(1 for r in rows if r.get('citations'))
print(f"rows={n}")
print(f"errors={len(errs)}")
print(f"total_citations={total_cits}")
print(f"gens_with_cits={gens_with_cits} ({100*gens_with_cits/n:.1f}%)")
print(f"mean_cits_per_gen={total_cits/n:.3f}")
print(f"mean_cits_when_present={total_cits/gens_with_cits:.2f}" if gens_with_cits else "no gens with cits")
