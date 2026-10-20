"""One-off: inspect blind handcheck CSV state before shuffling."""
import pandas as pd

df = pd.read_csv("data/handcheck/step0_blind_handcheck.csv")
print("Columns:", df.columns.tolist())
print("Rows:", len(df))
print()
print("First 20 rows — order check:")
print(df[["row_id", "domain", "cell"]].head(20).to_string())
print()
print("Cell distribution:")
print(df.groupby(["domain", "cell"]).size())
print()
print("Existing labels:")
print("  your_status filled:", df["your_status"].notna().sum())
print("  your_notes filled:", df["your_notes"].notna().sum())
