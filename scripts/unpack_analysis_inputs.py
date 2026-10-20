"""Gunzip every data/analysis_inputs/*/generations.jsonl.gz in place.

The analysis frames ship compressed because the uncompressed generations file is
close to GitHub's 100 MB per-file limit. Existing .jsonl files are left alone.
"""

import gzip
import shutil
from pathlib import Path

INPUTS_DIR = Path(__file__).resolve().parent.parent / "data" / "analysis_inputs"


def main() -> None:
    for packed in sorted(INPUTS_DIR.glob("*/generations.jsonl.gz")):
        target = packed.with_suffix("")
        if target.exists():
            print(f"exists, skipped: {target}")
            continue
        with gzip.open(packed, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"unpacked: {target}")


if __name__ == "__main__":
    main()
