"""Find non-ASCII chars in raw_span / output_excerpt to diagnose encoding."""
import io
import pandas as pd
import unicodedata
from collections import Counter

df = pd.read_csv("data/handcheck/step0_blind_handcheck.csv", encoding="utf-8")

text_cols = ["raw_span", "output_excerpt", "evidence_summary"]
counter = Counter()
samples = {}

for col in text_cols:
    for idx, val in df[col].items():
        if not isinstance(val, str):
            continue
        for ch in val:
            if ord(ch) > 127:
                counter[ch] += 1
                samples.setdefault(ch, (col, idx, val[:120]))

buf = io.StringIO()
buf.write(f"Distinct non-ASCII chars: {len(counter)}\n\n")
for ch, n in counter.most_common(60):
    name = unicodedata.name(ch, "<unnamed>")
    col, idx, sample = samples[ch]
    buf.write(f"  U+{ord(ch):04X}  n={n:4}  {name}\n")
    buf.write(f"    first seen: {col}[row {idx}] sample={sample!r}\n")

# Find rows with the replacement char
buf.write("\n=== Rows containing U+FFFD (corruption marker) ===\n")
for col in text_cols:
    for idx, val in df[col].items():
        if isinstance(val, str) and "�" in val:
            count = val.count("�")
            buf.write(f"  {col}[presentation_order={df.iloc[idx]['presentation_order']}, row_id={df.iloc[idx]['row_id']}] has {count} U+FFFD\n")
            window = val.replace("�", "[BAD]")
            buf.write(f"    {window[:200]!r}\n")

with open("scripts/_inspect_output.txt", "w", encoding="utf-8") as f:
    f.write(buf.getvalue())
print("Wrote scripts/_inspect_output.txt")
