"""Shuffle blind handcheck CSV row order with a fixed seed.

Anchoring-bias mitigation (Landis & Koch 1977): the original row order clusters
by domain (legal rows 1+4+5+10+11+12 together, academic 6-9 together) and opens
with Loving v. Virginia (unambiguous REAL), which primes confirmation bias.

Seed = 42 for reproducibility. The reveal CSV is keyed on row_id so it stays
joinable post-shuffle.
"""
import pandas as pd

SRC = "data/handcheck/step0_blind_handcheck.csv"
SEED = 42

df = pd.read_csv(SRC)
assert df["your_status"].isna().all(), "Refusing to shuffle a CSV with existing labels"
assert df["your_notes"].isna().all(), "Refusing to shuffle a CSV with existing labels"

shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Preserve original row_id (joins to verifier reveal). Add a presentation_order
# column for cosmetic numbering during the hand-check.
shuffled.insert(0, "presentation_order", range(1, len(shuffled) + 1))

shuffled.to_csv(SRC, index=False)

print(f"Shuffled {len(shuffled)} rows with seed={SEED}")
print()
print("New order — first 10:")
print(
    shuffled[["presentation_order", "row_id", "domain", "cell", "title", "case_name"]]
    .head(10)
    .to_string()
)
print()
print("Cell × domain distribution (unchanged):")
print(shuffled.groupby(["domain", "cell"]).size())
