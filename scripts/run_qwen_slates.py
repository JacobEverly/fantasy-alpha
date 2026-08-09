#!/usr/bin/env python3
"""Finish the BreakoutBench untrained-baseline run on PI serverless inference.

For each season slate missing a qwen_picks file: one chat request (temp 0),
strict-JSON parse with one repair retry. Then score every season via
evals/anon_demo.py and write the aggregate to evals/results/.
"""
from __future__ import annotations

import json
import re
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.pinference.ai/api/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
SEASONS = range(2015, 2025)

_ctx = ssl.create_default_context()
if not _ctx.cert_store_stats().get("x509_ca"):
    _ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")


def key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("PRIME_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no PRIME_API_KEY")


def chat(messages: list[dict]) -> tuple[str, dict]:
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 2000,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key()}", "Content-Type": "application/json",
                 # bill the team account, not the (empty) personal balance
                 "X-Prime-Team-ID": "cmskuc3x7018j7g10tr01gukl",
                 # the API's WAF rejects urllib's default Python-urllib UA
                 "User-Agent": "fantasy-alpha-evals/0.1"},
    )
    with urllib.request.urlopen(req, timeout=180, context=_ctx) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"], out.get("usage", {})


def parse_picks(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"\[.*\]", text, flags=re.S)
    picks = json.loads(m.group(0) if m else text)
    assert isinstance(picks, list) and len(picks) == 10, f"bad picks shape: {len(picks) if isinstance(picks, list) else type(picks)}"
    return [{"anon_id": str(p["anon_id"]), "p_breakout": float(p["p_breakout"])} for p in picks]


def main() -> int:
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for season in SEASONS:
        picks_path = ROOT / "evals" / "demo" / f"qwen_picks_{season}_ppr.json"
        if picks_path.exists():
            continue
        slate = json.loads((ROOT / "evals" / "demo" / f"slate_{season}_ppr.json").read_text())
        messages = [
            {"role": "system", "content": "You are a fantasy football analyst. Answer with strict JSON only."},
            {"role": "user", "content": slate["task"] + "\n\nCandidates:\n" + json.dumps(slate["candidates"])
             + '\n\nReturn a JSON array of exactly 10 objects: {"anon_id": str, "p_breakout": float 0-1}, most likely first. No prose.'},
        ]
        text, usage = chat(messages)
        try:
            picks = parse_picks(text)
        except Exception as e:
            messages += [{"role": "assistant", "content": text},
                         {"role": "user", "content": f"Invalid ({e}). Return ONLY the JSON array of exactly 10 objects."}]
            text, usage2 = chat(messages)
            usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in set(usage) | set(usage2)}
            picks = parse_picks(text)
        picks_path.write_text(json.dumps(picks, indent=1))
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)
        print(f"{season}: picks saved ({usage.get('prompt_tokens')}in/{usage.get('completion_tokens')}out)", flush=True)

    # score all seasons
    results, pooled = [], {"hits": 0, "expected": 0.0, "brier_sum": 0.0, "n": 0}
    for season in SEASONS:
        out = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "evals/anon_demo.py", "score", str(season), "ppr",
             f"evals/demo/qwen_picks_{season}_ppr.json"],
            capture_output=True, text=True, cwd=ROOT,
        ).stdout
        m = re.search(r"true breakouts among them: (\d+) \(base rate ([\d.]+)%; random@10 ≈ ([\d.]+)", out)
        h = re.search(r"model hits@10: (\d+).*Brier\(picks\): ([\d.]+)", out)
        n_break, rand10, hits, brier = int(m.group(1)), float(m.group(3)), int(h.group(1)), float(h.group(2))
        results.append((season, n_break, hits, rand10, brier))
        pooled["hits"] += hits
        pooled["expected"] += rand10
        pooled["brier_sum"] += brier * 10
        pooled["n"] += 10

    lines = ["# BreakoutBench — untrained open-model baseline (B2-naked)", "",
             f"Model: `{MODEL}` via Prime Intellect serverless (`api.pinference.ai`), temp 0, anonymized structure track.",
             "", "| season | breakouts in slate | hits@10 | random@10 | Brier |", "|---|---|---|---|---|"]
    for season, nb, hits, rand10, brier in results:
        lines.append(f"| {season} | {nb} | {hits} | {rand10:.1f} | {brier:.3f} |")
    lift = pooled["hits"] / pooled["expected"] if pooled["expected"] else 0
    lines += ["", f"**Pooled: {pooled['hits']} hits / {pooled['expected']:.1f} expected at random "
              f"(lift {lift:.1f}x) · mean Brier {pooled['brier_sum']/pooled['n']:.3f} · {pooled['n']} picks over 10 seasons**",
              "", "Comparison: Claude (Fable, in-session demo, 2019 only, partial de-anonymization disclosed): 3 hits, 4.0x lift, Brier 0.192.",
              f"", f"This run's marginal usage: {total_usage['prompt_tokens']}in/{total_usage['completion_tokens']}out tokens."]
    outdir = ROOT / "evals" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "qwen_breakout_baseline.md").write_text("\n".join(lines))
    print("\n".join(lines[4:]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
