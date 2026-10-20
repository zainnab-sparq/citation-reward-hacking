"""Pull per-message usage from the v2 judge batch and ground-truth per-citation cost."""
import asyncio
import io
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from anthropic import AsyncAnthropic

BATCH_ID = "msgbatch_011zianZJ5WDQwWjqTAuwpry"
BETA = "code-execution-web-tools-2026-02-09"

# Anthropic batched Sonnet 4.6 pricing (per memory + verified web search):
PRICE_IN_PER_M = 1.50  # batched input
PRICE_OUT_PER_M = 7.50  # batched output
SEARCH_FEE = 0.01  # $10/1000, flat (no batch discount)


async def main():
    client = AsyncAnthropic()
    it = await client.beta.messages.batches.results(BATCH_ID, betas=[BETA])
    rows = []
    async for item in it:
        rows.append(item)

    out = io.StringIO()
    n_ok = 0
    total_in = 0
    total_out = 0
    total_search = 0
    per_row = []
    for r in rows:
        if getattr(r.result, "type", None) != "succeeded":
            continue
        msg = r.result.message
        usage = getattr(msg, "usage", None)
        # Server-tool counts may live as a separate field in usage
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        # server tool counts: web_search_requests is the canonical key
        server_tool_use = getattr(usage, "server_tool_use", None)
        if server_tool_use is not None:
            search_n = getattr(server_tool_use, "web_search_requests", 0) or 0
        else:
            search_n = 0
        per_row.append((in_tok, out_tok, search_n, cache_read, cache_create))
        n_ok += 1
        total_in += in_tok
        total_out += out_tok
        total_search += search_n

    out.write(f"successful rows: {n_ok}\n")
    if n_ok == 0:
        Path("scripts/_usage.txt").write_text(out.getvalue(), encoding="utf-8")
        return
    out.write(f"total input tokens : {total_in:>10,}\n")
    out.write(f"total output tokens: {total_out:>10,}\n")
    out.write(f"total web searches : {total_search:>10,}\n\n")
    out.write(f"mean per-citation input : {total_in / n_ok:>8.0f} tokens\n")
    out.write(f"mean per-citation output: {total_out / n_ok:>8.0f} tokens\n")
    out.write(f"mean searches/citation  : {total_search / n_ok:>8.2f}\n\n")

    cost_in = total_in / 1_000_000 * PRICE_IN_PER_M
    cost_out = total_out / 1_000_000 * PRICE_OUT_PER_M
    cost_search = total_search * SEARCH_FEE
    total = cost_in + cost_out + cost_search
    out.write(f"batched input  cost: ${cost_in:.4f}\n")
    out.write(f"batched output cost: ${cost_out:.4f}\n")
    out.write(f"search fees        : ${cost_search:.4f}\n")
    out.write(f"-----------------------------\n")
    out.write(f"TOTAL Step 0 spend : ${total:.4f}  (n=50 citations)\n")
    out.write(f"per-citation cost  : ${total / n_ok:.4f}\n\n")

    # Scale projection
    out.write("=== Phase B projection at this actual per-citation cost ===\n")
    per = total / n_ok
    for label, n_cit in [
        ("Step 1 (Haiku 4.5) NO dedup", 36_000),
        ("Step 1 with 50% dedup", 18_000),
        ("Steps 1+3 NO dedup", 108_000),
        ("Steps 1+3 with 67% dedup", 36_000),
    ]:
        out.write(f"  {label:38} ${per * n_cit:>8,.0f}\n")

    Path("scripts/_usage.txt").write_text(out.getvalue(), encoding="utf-8")
    print("wrote scripts/_usage.txt")


asyncio.run(main())
