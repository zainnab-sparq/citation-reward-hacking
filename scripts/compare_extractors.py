"""Compare Haiku 4.5 vs Sonnet 4.6 as citation extractors on the pilot_b 60-gen set.

Computes per-(question_id, condition, trial_id) agreement on:
  - citation count (exact)
  - per-citation title (normalized, ordered-match within record)
  - per-citation first author surname (if any)
  - per-citation year
  - per-citation case_name / reporter (legal)

Outputs a summary block to stdout + a JSON file at data/extractions/pilot_b_haiku_extractor/comparison_summary.json.
Decision rule per the Phase B Path C plan: Haiku passes if per-citation field agreement >= 90%.
"""
import io
import json
import re
from pathlib import Path

SONNET = Path("data/extractions/pilot_b/claude-haiku-4-5-20251001.jsonl")
HAIKU = Path("data/extractions/pilot_b_haiku_extractor/claude-haiku-4-5-20251001.jsonl")
OUT_JSON = Path("data/extractions/pilot_b_haiku_extractor/comparison_summary.json")


def load_extractions(path):
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        key = (rec["question_id"], rec["condition"], int(rec.get("trial_id", 0)))
        out[key] = rec
    return out


def norm(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def best_match_pairs(s_cites, h_cites):
    """Pair Haiku citations to Sonnet citations by normalized-title overlap.

    Greedy: for each Sonnet citation, take the Haiku citation with the highest
    title token overlap (Jaccard). Used for per-field comparison only.
    Unmatched Sonnet citations count as missed; unmatched Haiku citations as extra.
    """
    pairs = []
    h_used = set()
    for si, sc in enumerate(s_cites):
        st = set(norm(sc.get("title", "")).split())
        if not st:
            st = set(norm(sc.get("case_name", "")).split())
        best_j = -1
        best_score = 0.0
        for hi, hc in enumerate(h_cites):
            if hi in h_used:
                continue
            ht = set(norm(hc.get("title", "")).split())
            if not ht:
                ht = set(norm(hc.get("case_name", "")).split())
            inter = len(st & ht)
            union = len(st | ht) or 1
            score = inter / union
            if score > best_score:
                best_score = score
                best_j = hi
        if best_j >= 0 and best_score >= 0.3:
            pairs.append((si, best_j, best_score))
            h_used.add(best_j)
        else:
            pairs.append((si, None, 0.0))
    extras = [hi for hi in range(len(h_cites)) if hi not in h_used]
    return pairs, extras


def field_agree(a, b):
    if a is None and b is None:
        return True
    return norm(str(a)) == norm(str(b))


def first_author_agree(sa, ha):
    sa_list = sa or []
    ha_list = ha or []
    if not sa_list and not ha_list:
        return True
    if not sa_list or not ha_list:
        return False
    return norm(sa_list[0]) == norm(ha_list[0])


def main():
    if not SONNET.exists():
        print(f"Sonnet baseline missing: {SONNET}")
        return
    if not HAIKU.exists():
        print(f"Haiku extractor output missing: {HAIKU}")
        return
    sonnet = load_extractions(SONNET)
    haiku = load_extractions(HAIKU)

    common = sorted(set(sonnet) & set(haiku))
    only_sonnet = sorted(set(sonnet) - set(haiku))
    only_haiku = sorted(set(haiku) - set(sonnet))

    n_records = len(common)
    n_count_agree = 0
    s_total_cites = 0
    h_total_cites = 0
    n_pairs_matched = 0
    n_pairs_total = 0
    n_title_agree = 0
    n_first_author_agree = 0
    n_year_agree = 0
    n_case_name_agree = 0
    n_legal_records = 0

    examples_disagree = []
    examples_count_mismatch = []

    for key in common:
        s_rec = sonnet[key]
        h_rec = haiku[key]
        s_cites = s_rec.get("citations") or []
        h_cites = h_rec.get("citations") or []
        s_total_cites += len(s_cites)
        h_total_cites += len(h_cites)
        if len(s_cites) == len(h_cites):
            n_count_agree += 1
        else:
            if len(examples_count_mismatch) < 5:
                examples_count_mismatch.append({
                    "key": list(key), "sonnet_n": len(s_cites), "haiku_n": len(h_cites),
                    "sonnet_titles": [c.get("title") or c.get("case_name") for c in s_cites],
                    "haiku_titles": [c.get("title") or c.get("case_name") for c in h_cites],
                })
        pairs, _extras = best_match_pairs(s_cites, h_cites)
        any_disagree_this_record = False
        for si, hi, score in pairs:
            n_pairs_total += 1
            if hi is None:
                any_disagree_this_record = True
                continue
            n_pairs_matched += 1
            sc = s_cites[si]; hc = h_cites[hi]
            title_ok = field_agree(sc.get("title"), hc.get("title"))
            author_ok = first_author_agree(sc.get("authors"), hc.get("authors"))
            year_ok = field_agree(sc.get("year"), hc.get("year"))
            case_ok = field_agree(sc.get("case_name"), hc.get("case_name"))
            n_title_agree += int(title_ok)
            n_first_author_agree += int(author_ok)
            n_year_agree += int(year_ok)
            if sc.get("case_name"):
                n_legal_records += 1
                n_case_name_agree += int(case_ok)
            if not (title_ok and author_ok and year_ok):
                any_disagree_this_record = True
        if any_disagree_this_record and len(examples_disagree) < 8:
            examples_disagree.append({
                "key": list(key),
                "sonnet_cites": s_cites,
                "haiku_cites": h_cites,
            })

    def pct(num, denom):
        return round(100.0 * num / denom, 1) if denom else 0.0

    summary = {
        "n_records_common": n_records,
        "n_records_only_sonnet": len(only_sonnet),
        "n_records_only_haiku": len(only_haiku),
        "sonnet_total_citations": s_total_cites,
        "haiku_total_citations": h_total_cites,
        "count_agreement_records_pct": pct(n_count_agree, n_records),
        "match_recall_pct": pct(n_pairs_matched, n_pairs_total),
        "title_agreement_among_matched_pct": pct(n_title_agree, n_pairs_matched),
        "first_author_agreement_among_matched_pct": pct(n_first_author_agree, n_pairs_matched),
        "year_agreement_among_matched_pct": pct(n_year_agree, n_pairs_matched),
        "case_name_agreement_among_legal_matched_pct": pct(n_case_name_agree, n_legal_records),
        "pass_threshold_pct": 90.0,
        "gate_pass_count_agreement": pct(n_count_agree, n_records) >= 90.0,
        "gate_pass_title_agreement": pct(n_title_agree, n_pairs_matched) >= 90.0,
        "gate_pass_match_recall": pct(n_pairs_matched, n_pairs_total) >= 90.0,
        "examples_count_mismatch_first_5": examples_count_mismatch,
        "examples_record_with_any_field_disagree_first_8": examples_disagree,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    buf = io.StringIO()
    buf.write("=" * 72 + "\n")
    buf.write("Haiku 4.5 vs Sonnet 4.6 extractor comparison (pilot_b 60-gen set)\n")
    buf.write("=" * 72 + "\n")
    buf.write(f"Records compared: {n_records}\n")
    buf.write(f"Sonnet-only keys: {len(only_sonnet)}  Haiku-only keys: {len(only_haiku)}\n")
    buf.write(f"Citation count totals: Sonnet={s_total_cites}  Haiku={h_total_cites}\n\n")
    buf.write(f"Count agreement (per-record exact n): {pct(n_count_agree, n_records)}% ({n_count_agree}/{n_records})\n")
    buf.write(f"Match recall (Sonnet cites paired to Haiku): {pct(n_pairs_matched, n_pairs_total)}% ({n_pairs_matched}/{n_pairs_total})\n")
    buf.write(f"Title agreement among matched: {pct(n_title_agree, n_pairs_matched)}%\n")
    buf.write(f"First-author agreement among matched: {pct(n_first_author_agree, n_pairs_matched)}%\n")
    buf.write(f"Year agreement among matched: {pct(n_year_agree, n_pairs_matched)}%\n")
    if n_legal_records:
        buf.write(f"Case-name agreement among legal-matched: {pct(n_case_name_agree, n_legal_records)}% ({n_case_name_agree}/{n_legal_records})\n")
    buf.write("\nGATE (>=90% on title agreement AND match recall AND count agreement):\n")
    buf.write(f"  title_agree>=90?      {summary['gate_pass_title_agreement']}\n")
    buf.write(f"  match_recall>=90?     {summary['gate_pass_match_recall']}\n")
    buf.write(f"  count_agree>=90?      {summary['gate_pass_count_agreement']}\n")
    buf.write("\nWrote " + str(OUT_JSON) + "\n")
    print(buf.getvalue())


if __name__ == "__main__":
    main()
