"""Re-write the blind handcheck CSV with a UTF-8 BOM so Excel-on-Windows
auto-detects the encoding instead of falling back to cp1252.

Pure encoding change: bytes after the BOM are byte-for-byte the original.
"""
from pathlib import Path

SRC = Path("data/handcheck/step0_blind_handcheck.csv")
BOM = b"\xef\xbb\xbf"

raw = SRC.read_bytes()

if raw.startswith(BOM):
    print("Already has UTF-8 BOM, nothing to do.")
else:
    SRC.write_bytes(BOM + raw)
    print(f"Prepended UTF-8 BOM to {SRC} ({len(raw)} -> {len(raw) + 3} bytes)")

verify = SRC.read_bytes()
assert verify.startswith(BOM)
print("Verified: file now starts with UTF-8 BOM.")
