import json
rows = [json.loads(l) for l in open('data/extractions/phase_b/extractions.jsonl', encoding='utf-8')]
no_title = no_raw_span = total = 0
sample_no_title = None
sample_no_raw_span = None
for r in rows:
    for c in r.get('citations', []):
        total += 1
        if 'title' not in c:
            no_title += 1
            if sample_no_title is None:
                sample_no_title = c
        if 'raw_span' not in c:
            no_raw_span += 1
            if sample_no_raw_span is None:
                sample_no_raw_span = c
print(f"total citations: {total}")
print(f"missing title: {no_title}")
print(f"missing raw_span: {no_raw_span}")
print("sample missing-title:", sample_no_title)
print("sample missing-raw_span:", sample_no_raw_span)
