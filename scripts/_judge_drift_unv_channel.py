"""Which coding of UNVERIFIABLE absorbs the judge's era-to-era movement?

At 31,980 judgments the three-era audit shows something the 28,266-row partial
could not: grounding fully recovers at block 48 (mean_sc 2.09 vs EARLY 2.03,
sc0 back under 1%) while UNVERIFIABLE keeps falling, 16.7% -> 13.8% -> 7.1%,
and FABRICATED never comes back down. So the residual RECOVERED-vs-EARLY gap is
not a REAL/FABRICATED boundary that moved. It is a judge that grew less willing
to abstain and spent those citations on FABRICATED.

That is testable directly. Score the same citations three ways and read off how
much era movement survives each:

  share_of_all   FABRICATED / all judged            (descriptive only)
  among_verif    FABRICATED / (FABRICATED + REAL)   registered primary coding,
                                                    UNVERIFIABLE NA-excluded
  unv_as_fab     (FABRICATED + UNVERIFIABLE) / all  registered Sensitivity #1,
                                                    excludes nothing

If `unv_as_fab` flattens the era gap that `among_verif` shows, the pre-registered
sensitivity analysis is drift-robust for the reason the D1 fit hinted at, and the
gap is a reporting-threshold shift rather than a verification failure.

Diagnostic only. Era and qtype carry no registered estimand: there is no stage
term and no condition term anywhere in this file.
"""
import collections
import json

PATH = "data/judgments/phase_c/claude-subagent-sonnet.jsonl"
STATUS_KEY = "judge_status"

DEGRADED_START = 33 * 500   # 16,500
RECOVERED_START = 48 * 500  # 24,000

ERA_ORDER = ("EARLY", "DEGRADED", "RECOVERED")


def _qtype(custom_id):
    question_id = custom_id.split("__")[0]
    return question_id.rsplit("_", 1)[0].split("_", 1)[1]


def _codings(chunk):
    """The three fabrication rates, in percent, for one set of judgments."""
    status = collections.Counter(row.get(STATUS_KEY) for row in chunk)
    real, fab, unv = status["REAL"], status["FABRICATED"], status["UNVERIFIABLE"]
    verifiable = real + fab
    return {
        "n": len(chunk),
        "share_of_all": 100 * fab / len(chunk),
        "among_verif": 100 * fab / verifiable if verifiable else float("nan"),
        "unv_as_fab": 100 * (fab + unv) / len(chunk),
        "unv_share": 100 * unv / len(chunk),
    }


def _print_block(title, groups):
    print(title)
    print(f"  {'era':<10} {'n':>6} {'UNV%':>7} {'shr_all':>8} "
          f"{'amg_ver':>8} {'unv=fab':>8}")
    for era in ERA_ORDER:
        r = _codings(groups[era])
        print(f"  {era:<10} {r['n']:>6} {r['unv_share']:>6.1f}% "
              f"{r['share_of_all']:>7.1f}% {r['among_verif']:>7.1f}% "
              f"{r['unv_as_fab']:>7.1f}%")
    early, recovered = _codings(groups["EARLY"]), _codings(groups["RECOVERED"])
    print(f"  {'R-E gap':<10} {'':>6} "
          f"{recovered['unv_share'] - early['unv_share']:>+6.1f}pp "
          f"{recovered['share_of_all'] - early['share_of_all']:>+7.1f}pp "
          f"{recovered['among_verif'] - early['among_verif']:>+7.1f}pp "
          f"{recovered['unv_as_fab'] - early['unv_as_fab']:>+7.1f}pp")
    print()


rows = [json.loads(line) for line in open(PATH, encoding="utf-8")]
for index, row in enumerate(rows):
    row["_era"] = ("EARLY" if index < DEGRADED_START
                   else "DEGRADED" if index < RECOVERED_START
                   else "RECOVERED")

by_era = collections.defaultdict(list)
for row in rows:
    by_era[row["_era"]].append(row)
_print_block("=== all judged citations ===", by_era)

by_qtype_era = collections.defaultdict(lambda: collections.defaultdict(list))
for row in rows:
    by_qtype_era[_qtype(row["custom_id"])][row["_era"]].append(row)
for qtype in ("known_real", "obscure_real", "unanswerable"):
    _print_block(f"=== qtype = {qtype} ===", by_qtype_era[qtype])

print("Read the two right-hand columns against each other. A large R-E gap under")
print("among_verif that shrinks under unv=fab means the era movement lived in the")
print("abstention threshold, not in the REAL-vs-FABRICATED boundary.")
