"""One-off: drop transient-error rows from the J2a production JSONL so the
resume pass picks them up. Successful rows (error=None) are kept verbatim."""
import json
from pathlib import Path

SRC = Path("data/judgments/phase_b/claude-subagent-sonnet.jsonl")


def main():
    rows = [json.loads(l) for l in SRC.read_text(encoding="utf-8").splitlines() if l.strip()]
    keep = [r for r in rows if r.get("error") is None]
    drop = [r for r in rows if r.get("error") is not None]
    print(f"total: {len(rows)}, keep (success): {len(keep)}, drop (transient): {len(drop)}")
    with SRC.open("w", encoding="utf-8") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    print(f"rewrote {SRC}")


if __name__ == "__main__":
    main()
