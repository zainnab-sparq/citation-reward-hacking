"""Write 2 real remaining-input prompt files to a non-Dropbox cache dir and
print a JSON args blob for the file-Read workflow smoke test."""
import json
from pathlib import Path

from src.judge_via_subagents import build_judge_inputs

EXTRACTIONS = Path("data/extractions/phase_b/extractions.jsonl")
VERIFICATIONS = Path("data/verifications/phase_b/verifications.jsonl")
QUESTIONS = Path("data/questions/frozen_900q_legal_amended.jsonl")

CACHE = Path.home() / ".cache" / "citation-judge-prompts"


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    picked = []
    for inp in build_judge_inputs(EXTRACTIONS, VERIFICATIONS, QUESTIONS):
        # one FAB, one UNV, both legal/academic for quick web checks
        if inp.verifier_status == "FABRICATED" and len([p for p in picked if p[2] == "FABRICATED"]) == 0:
            picked.append((inp.custom_id, inp.prompt, inp.verifier_status))
        elif inp.verifier_status == "UNVERIFIABLE" and len([p for p in picked if p[2] == "UNVERIFIABLE"]) == 0:
            picked.append((inp.custom_id, inp.prompt, inp.verifier_status))
        if len(picked) == 2:
            break

    items = []
    for custom_id, prompt, status in picked:
        path = CACHE / f"{custom_id}.txt"
        path.write_text(prompt, encoding="utf-8")
        items.append({"custom_id": custom_id, "prompt_path": str(path)})
        print(f"wrote {path}  ({status})")

    args_blob = {"items": items}
    Path("scripts/_read_smoke_args.json").write_text(json.dumps(args_blob, indent=2), encoding="utf-8")
    print("\n=== ARGS BLOB ===")
    print(json.dumps(args_blob, indent=2))


if __name__ == "__main__":
    main()
