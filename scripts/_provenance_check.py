import json

try:
    import scipy
    print("scipy", scipy.__version__)
except ImportError:
    print("scipy NOT available")

bak = "data/judgments/phase_b/claude-subagent-sonnet.jsonl.before_repurge_20260608_041032.bak"
PLACEHOLDER = "subprocess_failed_or_unparseable"

sub = wf = 0
for line in open(bak, encoding="utf-8"):
    s = line.strip()
    if not s:
        continue
    r = json.loads(s)
    if r.get("error") == PLACEHOLDER:
        wf += 1
    else:
        sub += 1
print("subprocess (real verdict in bak):", sub)
print("workflow   (placeholder in bak) :", wf)
