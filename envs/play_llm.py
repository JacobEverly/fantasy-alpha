#!/usr/bin/env python3
"""Drive an untrained open model through DraftGym episodes (B2-naked baseline).

12 episodes: seasons {2018, 2021, 2024} × slots {1, 7} × seeds {0, 1},
12-team PPR, 15 rounds. Model: Qwen/Qwen3.5-9B on Prime Intellect serverless
(same client pattern as scripts/run_qwen_slates.py: team billing header +
custom UA, temp 0, thinking off). The prompt exposes BOTH action types — a
pick or an evidence tool call — so we can measure whether the untrained model
uses the lookups at all.

Failure ladder per model turn: strict-JSON parse → one repair retry →
autopick fallback (counted). A per-pick request cap stops tool-loop runaways.

HARD BUDGET: $1.50 (serverless token rates below). The script stops issuing
requests and finishes remaining picks by autopick if the cap is near. Every
run appends its cost to docs/budget-ledger.md.
"""
from __future__ import annotations

import json
import re
import ssl
import statistics
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from evals.draftbench import _board_key
from envs.draftgym import DraftGym, Observation

ROOT = Path(__file__).resolve().parent.parent
API = "https://api.pinference.ai/api/v1/chat/completions"
MODEL = "Qwen/Qwen3.5-9B"
TEAM_ID = "cmskuc3x7018j7g10tr01gukl"

# Serverless rates for Qwen3.5-9B (docs/budget-ledger.md).
USD_PER_M_IN = 0.18
USD_PER_M_OUT = 0.54
BUDGET_USD = 1.50
MAX_REQUESTS_PER_PICK = 8  # tool cap (5) + repair headroom; then autopick

EPISODES = [
    dict(season=season, preset="ppr", agent_slot=slot, seed=seed)
    for season in (2018, 2021, 2024)
    for slot in (1, 7)
    for seed in (0, 1)
]
# Anonymized-mode duplicates of two named configs: paired named-minus-masked
# reward is a direct memorization measurement (risk register #8).
MASKED_EPISODES = [
    dict(season=2021, preset="ppr", agent_slot=1, seed=0, mask_names=True),
    dict(season=2024, preset="ppr", agent_slot=7, seed=0, mask_names=True),
]

_ctx = ssl.create_default_context()
if not _ctx.cert_store_stats().get("x509_ca"):
    _ctx = ssl.create_default_context(cafile="/etc/ssl/cert.pem")


def _key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        if line.startswith("PRIME_API_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit("no PRIME_API_KEY in .env")


def chat(messages: list[dict], max_tokens: int = 700) -> tuple[str, dict]:
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        API,
        data=json.dumps(body).encode(),
        headers={
            "Authorization": f"Bearer {_key()}",
            "Content-Type": "application/json",
            # bill the team account, not the (empty) personal balance
            "X-Prime-Team-ID": TEAM_ID,
            # the API's WAF rejects urllib's default Python-urllib UA
            "User-Agent": "fantasy-alpha-evals/0.1",
        },
    )
    with urllib.request.urlopen(req, timeout=180, context=_ctx) as resp:
        out = json.loads(resp.read())
    return out["choices"][0]["message"]["content"], out.get("usage", {})


# ---------------------------------------------------------------------------
# Prompting + action parsing (shared with envs/verifiers_v1_adapter.py)


SYSTEM_PROMPT = (
    "You are a fantasy football draft analyst on the clock in a snake draft. "
    "Respond with STRICT JSON only — a single object, no prose, no markdown.\n"
    'Either make your pick:  {"pick": "<player_id>"}\n'
    'Or look up evidence first:  {"tool": {"name": "<tool>", "arguments": {...}}}\n'
    "Available evidence tools (dated news archive — nothing after the draft "
    "date exists):\n"
    '  player_news   args {"player": "<name or id>", "limit": int}\n'
    '  search        args {"query": "<keywords>", "limit": int}\n'
    '  depth_chart   args {"team_or_player": "<TEAM or name>"}\n'
    '  injury_status args {"player": "<name>"}\n'
    "Lookups are capped per pick (see lookups_remaining). Pick player_id "
    "values must come from top_available."
)


MASKED_NOTE = (
    "\nThis draft is ANONYMIZED: players are shown as board ids (B001...) with "
    "real features. Only depth_chart and injury_status work, and they take a "
    "board id (e.g. {\"player\": \"B012\"}). player_news/search are disabled. "
    "Base your picks on the features, not on guessing identities."
)


def build_messages(obs: Observation) -> list[dict]:
    system = SYSTEM_PROMPT + (MASKED_NOTE if obs.get("anonymized") else "")
    user = (
        "Draft state (JSON):\n"
        + obs.render_json()
        + "\n\nRespond with ONE JSON action object now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _first_json_object(text: str) -> str | None:
    """Extract the first balanced {...} block (models love to wrap in prose)."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None


def parse_action(text: str) -> dict:
    """Strict parse of a model reply into a DraftGym action. Raises ValueError."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    blob = _first_json_object(text)
    if blob is None:
        raise ValueError("no JSON object in reply")
    obj = json.loads(blob)
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    if "pick" in obj:
        return {"pick": str(obj["pick"]), **(
            {"rationale": str(obj["rationale"])} if obj.get("rationale") else {}
        )}
    if "tool" in obj and isinstance(obj["tool"], dict):
        tool = obj["tool"]
        if "name" not in tool:
            raise ValueError("tool call missing 'name'")
        return {"tool": {"name": tool["name"], "arguments": tool.get("arguments", {})}}
    raise ValueError("object has neither 'pick' nor 'tool'")


def autopick_action(gym: DraftGym) -> dict:
    """The AutopickADP fallback: lowest-ADP feasible player right now."""
    return {"pick": min(gym._agent_candidates(), key=_board_key).player_id}


# ---------------------------------------------------------------------------
# Episode runner


class Budget:
    def __init__(self, cap_usd: float = BUDGET_USD):
        self.cap = cap_usd
        self.tokens_in = 0
        self.tokens_out = 0

    def add(self, usage: dict) -> None:
        self.tokens_in += usage.get("prompt_tokens", 0)
        self.tokens_out += usage.get("completion_tokens", 0)

    @property
    def spent(self) -> float:
        return (
            self.tokens_in * USD_PER_M_IN + self.tokens_out * USD_PER_M_OUT
        ) / 1_000_000

    @property
    def exhausted(self) -> bool:
        return self.spent >= self.cap * 0.95  # stop with 5% headroom


def run_episode(spec: dict, budget: Budget, verbose: bool = True) -> dict:
    gym = DraftGym(**spec)
    obs = gym.reset()
    stats = {
        **spec,
        "model_requests": 0,
        "fallback_picks": 0,
        "parse_failures": 0,
        "illegal_picks": 0,
        "tool_calls": 0,
        "tool_calls_ok": 0,
        "budget_truncated": False,
        "picks": [],
    }
    reward, done, info = 0.0, False, {}
    requests_this_pick = 0

    while not done:
        if budget.exhausted or requests_this_pick >= MAX_REQUESTS_PER_PICK:
            if budget.exhausted:
                stats["budget_truncated"] = True
            action = autopick_action(gym)
            stats["fallback_picks"] += 1
        else:
            action = None
            messages = build_messages(obs)
            for attempt in range(2):  # initial + one repair retry
                try:
                    text, usage = chat(messages)
                    budget.add(usage)
                    stats["model_requests"] += 1
                    requests_this_pick += 1
                    action = parse_action(text)
                    break
                except ValueError as e:
                    stats["parse_failures"] += 1
                    if attempt == 0:
                        messages = messages + [
                            {"role": "assistant", "content": text},
                            {
                                "role": "user",
                                "content": f"Invalid ({e}). Reply with ONLY one JSON "
                                'action object: {"pick": "<player_id>"} or '
                                '{"tool": {"name": ..., "arguments": {...}}}.',
                            },
                        ]
            if action is None:
                action = autopick_action(gym)
                stats["fallback_picks"] += 1

        if "tool" in action:
            stats["tool_calls"] += 1
            obs, reward, done, info = gym.step(action)
            if obs["tool_results"] and obs["tool_results"][-1]["response"].get("ok"):
                stats["tool_calls_ok"] += 1
            continue

        was_pick_number = obs["pick_number"]
        try:
            obs, reward, done, info = gym.step(action)
        except ValueError:  # illegal pick -> autopick fallback
            stats["illegal_picks"] += 1
            stats["fallback_picks"] += 1
            obs, reward, done, info = gym.step(autopick_action(gym))
        stats["picks"].append({"pick_number": was_pick_number, "action": action})
        requests_this_pick = 0

    stats["reward"] = reward
    stats["agent_points_realistic"] = info.get("agent_points_realistic")
    stats["control_points_realistic"] = info.get("control_points_realistic")
    stats["total_lookups"] = info.get("total_lookups", 0)
    stats["cap_hits"] = info.get("cap_hits", 0)
    if verbose:
        print(
            f"  {spec['season']} slot {spec['agent_slot']} seed {spec['seed']}: "
            f"reward {reward:+.1f}  fallbacks {stats['fallback_picks']}/15  "
            f"tools {stats['tool_calls']}  spent ${budget.spent:.3f}",
            flush=True,
        )
    return stats


def append_ledger(budget: Budget, n_episodes: int) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    line = (
        f"| {stamp} | DraftGym untrained-baseline: {n_episodes} episodes "
        f"(2018/2021/2024 × slots 1,7 × 2 seeds, 12-team ppr, 15 rounds), "
        f"Qwen/Qwen3.5-9B temp 0, pick-or-tool prompt (envs/play_llm.py) "
        f"| ${USD_PER_M_IN}/M in, ${USD_PER_M_OUT}/M out "
        f"| {budget.tokens_in:,} in / {budget.tokens_out:,} out tok "
        f"| ~${budget.spent:.3f} |"
    )
    path = ROOT / "docs" / "budget-ledger.md"
    text = path.read_text().rstrip("\n")
    # keep the running-total line last
    lines = text.splitlines()
    total_idx = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Total to date")), None
    )
    if total_idx is not None:
        prior = float(re.search(r"\$([\d.]+)", lines[total_idx]).group(1))
        lines[total_idx] = f"Total to date: ~${prior + budget.spent:.3f}"
        lines.insert(total_idx - 1 if lines[total_idx - 1] == "" else total_idx, line)
        # ensure blank line before total
        path.write_text("\n".join(lines) + "\n")
    else:
        path.write_text(text + "\n" + line + "\n")


def main() -> int:
    budget = Budget()
    results = []
    all_specs = EPISODES + MASKED_EPISODES
    print(
        f"DraftGym × {MODEL} — {len(EPISODES)} named + "
        f"{len(MASKED_EPISODES)} anonymized episodes, budget ${BUDGET_USD:.2f}"
    )
    for spec in all_specs:
        results.append(run_episode(spec, budget))

    named = [r for r in results if not r.get("mask_names")]
    masked = [r for r in results if r.get("mask_names")]
    # paired named-minus-masked delta on identical (season, slot, seed) configs
    deltas = []
    for m in masked:
        twin = next(
            (n for n in named
             if (n["season"], n["agent_slot"], n["seed"]) == (m["season"], m["agent_slot"], m["seed"])),
            None,
        )
        if twin:
            deltas.append(round(twin["reward"] - m["reward"], 2))

    rewards = [r["reward"] for r in named]
    picks_total = sum(len(r["picks"]) + r["illegal_picks"] for r in results)
    fallbacks = sum(r["fallback_picks"] for r in results)
    tool_calls = sum(r["tool_calls"] for r in results)
    summary = {
        "model": MODEL,
        "episodes": len(results),
        "named_episodes": len(named),
        "mean_reward": round(statistics.mean(rewards), 2),
        "stdev_reward": round(statistics.stdev(rewards), 2) if len(rewards) > 1 else 0.0,
        "rewards": rewards,
        "masked_episodes": len(masked),
        "masked_rewards": [r["reward"] for r in masked],
        "named_minus_masked_paired_deltas": deltas,
        "masked_tool_calls": sum(r["tool_calls"] for r in masked),
        "fallback_picks": fallbacks,
        "fallback_rate": round(fallbacks / max(1, picks_total), 4),
        "illegal_picks": sum(r["illegal_picks"] for r in results),
        "parse_failures": sum(r["parse_failures"] for r in results),
        "tool_calls": tool_calls,
        "tool_calls_ok": sum(r["tool_calls_ok"] for r in results),
        "tool_calls_per_pick": round(tool_calls / max(1, picks_total), 3),
        "episodes_using_tools": sum(1 for r in results if r["tool_calls"] > 0),
        "budget_truncated_episodes": sum(1 for r in results if r["budget_truncated"]),
        "tokens_in": budget.tokens_in,
        "tokens_out": budget.tokens_out,
        "est_cost_usd": round(budget.spent, 4),
    }
    out = ROOT / "evals" / "results" / "draftgym_qwen_baseline.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "episodes": results}, indent=1))
    append_ledger(budget, len(results))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
