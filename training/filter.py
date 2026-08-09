"""Survivor-bias-proof SFT trace filter (docs/training-risk-register.md risk 3).

Pure functions, no I/O. Implements the four pinned rules:

(a) **Process gate (hard):** every quantitative claim in the assistant trace
    must appear in the tool-provided context (packet + grounding + evidence,
    i.e. the user message). Numbers absent from tool outputs -> drop
    (CLAUDE.md hard rule: the LLM never invents numbers).
(b) **Grounding gate (hard):** the trace's final probability must fall within
    the plausibility band [anchor/3, min(0.95, anchor*5)] of the
    code-computed historical grounding anchor. Kills lucky-90% traces
    regardless of outcome. (The risk-register allows an evidence-typed
    exception for larger deviations; v0 implements the strict band and logs
    the drops — the exception path is deliberately NOT implemented yet.)
(c) **Calibration score, never hit/miss:** among gate survivors, score each
    trace by Brier improvement vs the grounding anchor
    ((anchor - y)^2 - (p - y)^2) and keep the top ``keep_frac`` (~60%).
    The cut is applied WITHIN (stated-probability-band x outcome) strata:
    raw global Brier ranking still favors hits inside any given band (a
    p=0.65 hit outscores a p=0.65 miss by construction), which is precisely
    the survivorship channel rule (d) polices — the first pilot run flagged
    it empirically. Stratifying makes inclusion conditionally independent
    of the outcome given the stated probability, so rule (d) holds by
    construction; within a stratum the Brier score still ranks anchor
    discipline. Well-reasoned misses survive.
(d) **Survivorship audit:** per stated-probability band, the kept-trace hit
    rate must be statistically indistinguishable from the pre-filter hit
    rate (a kept-set hit-rate jump = survivorship leaked back in). Emitted
    with every filter run.

A trace dict needs: ``assistant`` (str), ``user`` (str), ``outcome`` (0/1),
``anchor_rate`` (float|None), and ideally ``trace_id``. The final probability
is parsed from the assistant text's ``FINAL PROBABILITY:`` line.

Stdlib only. Python 3.11+.
"""
from __future__ import annotations

import math
import re
from typing import Mapping, Sequence

FINAL_P_RE = re.compile(r"FINAL PROBABILITY:\s*([01](?:\.\d+)?)", re.IGNORECASE)
# Numbers not embedded in a word/identifier; optional trailing %.
NUMBER_RE = re.compile(r"(?<![\w.\-])(\d+(?:\.\d+)?)(\s*%)?")

# Prose-safe small integers ("3 of his last 4 games", "two of ten") — counts,
# not quantitative claims about a data source. Documented exemption.
SMALL_INT_EXEMPT_MAX = 10

BAND_EDGES = tuple(i / 10 for i in range(11))  # deciles of stated probability
DEFAULT_KEEP_FRAC = 0.6
ABS_TOL = 0.01   # |x - c| tolerance against a context number
REL_TOL = 0.02   # relative tolerance (rounded prose like "about 21.5" vs 21.54)


# ---------------------------------------------------------------------------
# parsing


def parse_final_p(assistant: str) -> float | None:
    """Final probability from the trace's FINAL PROBABILITY line (last wins)."""
    matches = FINAL_P_RE.findall(assistant or "")
    if not matches:
        return None
    p = float(matches[-1])
    return p if 0.0 <= p <= 1.0 else None


def extract_numbers(text: str) -> list[tuple[float, str, bool]]:
    """All numeric literals in ``text`` as (value, raw, is_percent)."""
    out = []
    for m in NUMBER_RE.finditer(text or ""):
        raw = m.group(0).strip()
        out.append((float(m.group(1)), raw, m.group(2) is not None))
    return out


CONTEXT_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def context_number_set(user: str) -> set[float]:
    """Every number present in the tool-provided context (the user message).

    Uses a laxer regex than the claim extractor so date components count
    ("2016-08-30" grounds a prose "August 30"), and adds y-1/y-2 for every
    year-like value so "his 2015 season" is citable when the as-of date says
    2016 and the packet carries prior_season/two_seasons_ago blocks — those
    are inferences from the context, not invented numbers.
    """
    ctx = {float(m) for m in CONTEXT_NUMBER_RE.findall(user or "")}
    for v in list(ctx):
        if v == int(v) and 1900 <= v <= 2100:
            ctx |= {v - 1, v - 2}
    return ctx


# ---------------------------------------------------------------------------
# rule (a): cite-or-don't-claim


def _matches(x: float, ctx: set[float]) -> bool:
    for c in ctx:
        if abs(x - c) <= ABS_TOL:
            return True
        if c != 0 and abs(x - c) / abs(c) <= REL_TOL:
            return True
    return False


def uncited_numbers(trace: Mapping) -> list[str]:
    """Numbers in the assistant text with no source in the context.

    A number is grounded if it (or its /100 / x100 percent rescaling) matches
    a context number within tolerance, equals the trace's own final
    probability (the one number the trace is *producing*, not citing), or is
    a prose-safe integer <= SMALL_INT_EXEMPT_MAX.
    """
    assistant = trace.get("assistant") or ""
    ctx = context_number_set(trace.get("user") or "")
    p = parse_final_p(assistant)
    if p is not None:
        ctx |= {p, round(p * 100, 6)}
    body = FINAL_P_RE.sub("", assistant)  # the FINAL line is exempt
    bad = []
    for value, raw, is_percent in extract_numbers(body):
        if value == int(value) and 0 <= value <= SMALL_INT_EXEMPT_MAX and not is_percent:
            continue
        candidates = [value, value / 100, value * 100]
        if any(_matches(c, ctx) for c in candidates):
            continue
        bad.append(raw)
    return bad


# ---------------------------------------------------------------------------
# rule (b): grounding plausibility band


def grounding_band(anchor_rate: float) -> tuple[float, float]:
    """[anchor/3, min(0.95, anchor*5)], anchor floored at 0.02 so a zero-rate
    cohort still yields a usable band."""
    a = max(float(anchor_rate), 0.02)
    return (a / 3, min(0.95, a * 5))


def within_band(p: float, anchor_rate: float) -> bool:
    lo, hi = grounding_band(anchor_rate)
    return lo <= p <= hi


# ---------------------------------------------------------------------------
# rule (c): Brier improvement vs the grounding anchor


def brier_improvement(p: float, anchor_rate: float, outcome: int) -> float:
    """(anchor - y)^2 - (p - y)^2: positive = beat the base-rate forecaster."""
    y = int(outcome)
    return (float(anchor_rate) - y) ** 2 - (float(p) - y) ** 2


# ---------------------------------------------------------------------------
# rule (d): survivorship audit


def band_of(p: float) -> tuple[float, float]:
    for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
        if lo <= p < hi or (hi == 1.0 and p == 1.0):
            return (lo, hi)
    raise ValueError(f"probability out of range: {p}")


def band_audit(pre_traces: Sequence[Mapping],
               kept_traces: Sequence[Mapping]) -> list[dict]:
    """Kept vs pre-filter hit rates per stated-probability decile.

    ``pre_traces`` = every trace with a parseable probability (before any
    filtering); ``kept_traces`` = the filter's survivors. Flags a band when
    the kept hit rate departs from the pre-filter hit rate by more than two
    binomial standard errors (heuristic two-sigma check at kept-set n) —
    a kept-set hit-rate JUMP is the survivor-bias signature.
    """
    kept_ids = {id(t) for t in kept_traces}  # identity is fine: same objects
    by_band: dict[tuple[float, float], dict] = {}
    for t in pre_traces:
        p = t.get("p")
        if p is None:
            p = parse_final_p(t.get("assistant") or "")
        if p is None:
            continue
        b = band_of(p)
        row = by_band.setdefault(b, {"n_pre": 0, "hits_pre": 0,
                                     "n_kept": 0, "hits_kept": 0})
        row["n_pre"] += 1
        row["hits_pre"] += int(t["outcome"])
        if id(t) in kept_ids:
            row["n_kept"] += 1
            row["hits_kept"] += int(t["outcome"])
    rows = []
    for (lo, hi), r in sorted(by_band.items()):
        pre_rate = r["hits_pre"] / r["n_pre"] if r["n_pre"] else None
        kept_rate = r["hits_kept"] / r["n_kept"] if r["n_kept"] else None
        diff = se = None
        flagged = False
        if pre_rate is not None and kept_rate is not None and r["n_kept"] > 0:
            diff = kept_rate - pre_rate
            se = math.sqrt(max(pre_rate * (1 - pre_rate), 1e-9) / r["n_kept"])
            flagged = r["n_kept"] >= 5 and abs(diff) > 2 * se
        rows.append({"band": (lo, hi), **r, "pre_hit_rate": pre_rate,
                     "kept_hit_rate": kept_rate, "diff": diff, "se": se,
                     "flagged": flagged})
    return rows


# ---------------------------------------------------------------------------
# orchestration


def run_filter(traces: Sequence[Mapping],
               keep_frac: float = DEFAULT_KEEP_FRAC) -> dict:
    """Apply rules (a)-(d). Returns kept traces (annotated with
    ``p``/``brier_improvement``), rejections with reasons, the reason
    histogram, and the survivorship audit table."""
    rejected: list[tuple[Mapping, str]] = []
    survivors: list[dict] = []
    parseable: list[Mapping] = []

    for t in traces:
        p = parse_final_p(t.get("assistant") or "")
        if p is None:
            rejected.append((t, "unparseable_probability"))
            continue
        t = dict(t)
        t["p"] = p
        parseable.append(t)
        bad = uncited_numbers(t)
        if bad:
            rejected.append((t, f"uncited_number: {', '.join(bad[:5])}"))
            continue
        anchor = t.get("anchor_rate")
        if anchor is None:
            rejected.append((t, "no_grounding_anchor"))
            continue
        if not within_band(p, anchor):
            lo, hi = grounding_band(anchor)
            rejected.append(
                (t, f"grounding_band: p={p:.2f} outside [{lo:.3f}, {hi:.3f}] "
                    f"(anchor {anchor:.3f})"))
            continue
        t["brier_improvement"] = brier_improvement(p, anchor, t["outcome"])
        survivors.append(t)

    # rule (c): keep the top keep_frac by Brier-improvement-vs-base-rate,
    # stratified by (stated-probability band x outcome) so the cut cannot
    # shift per-band hit rates (rule d holds by construction).
    strata: dict[tuple, list[dict]] = {}
    for t in survivors:
        strata.setdefault((band_of(t["p"]), int(t["outcome"])), []).append(t)
    kept: list[dict] = []
    for cell in strata.values():
        cell.sort(key=lambda t: (-t["brier_improvement"],
                                 str(t.get("trace_id", ""))))
        n_keep = int(keep_frac * len(cell) + 0.5)  # half-up: singletons keep 1
        kept.extend(cell[:n_keep])
        for t in cell[n_keep:]:
            rejected.append(
                (t, f"brier_cutoff: improvement {t['brier_improvement']:+.4f} "
                    f"below top-{keep_frac:.0%} of its (p-band, outcome) "
                    "stratum"))
    kept.sort(key=lambda t: str(t.get("trace_id", "")))

    reason_counts: dict[str, int] = {}
    for _, reason in rejected:
        key = reason.split(":", 1)[0]
        reason_counts[key] = reason_counts.get(key, 0) + 1

    return {
        "n_input": len(traces),
        "n_parseable": len(parseable),
        "kept": kept,
        "rejected": rejected,
        "reason_counts": reason_counts,
        "keep_frac": keep_frac,
        "band_audit": band_audit(parseable, kept),
    }


# ---------------------------------------------------------------------------
# report


def _fmt_rate(x: float | None) -> str:
    return f"{x:.3f}" if x is not None else "—"


def render_report(result: Mapping, title: str = "SFT pilot v0 — filter report",
                  n_examples: int = 3) -> str:
    """Markdown report: kept %, rejection histogram, band-audit table, and
    verbatim example traces (kept + rejected)."""
    n_in = result["n_input"]
    kept = result["kept"]
    lines = [f"# {title}", ""]
    lines.append(f"Input traces: **{n_in}** · parseable probability: "
                 f"{result['n_parseable']} · kept: **{len(kept)}** "
                 f"({len(kept) / n_in:.1%} of input)" if n_in else "No input traces.")
    lines += ["",
              "Filter rules (docs/training-risk-register.md risk 3): "
              "(a) every quantitative claim cites a tool output; "
              "(b) final probability within [anchor/3, min(0.95, anchor×5)] "
              "of the code-computed grounding anchor; "
              f"(c) keep top {result['keep_frac']:.0%} of gate survivors by "
              "Brier improvement vs the grounding base rate, within "
              "(probability-band × outcome) strata — never binary hits; "
              "(d) survivorship audit below.", ""]

    lines += ["## Rejection reasons", ""]
    if result["rejected"]:
        lines += ["| reason | n |", "|---|---|"]
        for reason, n in sorted(result["reason_counts"].items(),
                                key=lambda kv: -kv[1]):
            lines.append(f"| {reason} | {n} |")
    else:
        lines.append("No rejections.")

    lines += ["", "## Survivorship audit (rule d)", "",
              "Kept-trace hit rates must match pre-filter hit rates within "
              "each stated-probability band; a kept-set jump means survivor "
              "bias leaked back in. Flag = |diff| > 2·SE at kept-set n "
              "(bands with n_kept ≥ 5).", "",
              "| p band | n pre | hit rate pre | n kept | hit rate kept | diff | flag |",
              "|---|---|---|---|---|---|---|"]
    for r in result["band_audit"]:
        lo, hi = r["band"]
        diff = f"{r['diff']:+.3f}" if r["diff"] is not None else "—"
        flag = "**FLAG**" if r["flagged"] else ""
        lines.append(f"| [{lo:.1f}, {hi:.1f}) | {r['n_pre']} | "
                     f"{_fmt_rate(r['pre_hit_rate'])} | {r['n_kept']} | "
                     f"{_fmt_rate(r['kept_hit_rate'])} | {diff} | {flag} |")
    if any(r["flagged"] for r in result["band_audit"]):
        lines += ["", "**At least one band is flagged — inspect before "
                      "training touches this corpus.**"]
    else:
        lines += ["", "No band flagged: kept hit rates are consistent with "
                      "pre-filter hit rates (no survivorship signature)."]

    def _example(t: Mapping, reason: str | None) -> list[str]:
        meta = (f"trace `{t.get('trace_id', '?')}` · {t.get('bench', '?')}/"
                f"{t.get('family', '?')} · season {t.get('season', '?')} · "
                f"player {t.get('player', '?')} · p={t.get('p')} · "
                f"outcome={t.get('outcome')} · anchor={t.get('anchor_rate')}")
        out = ["", f"### {meta}", ""]
        if reason:
            out += [f"**Rejected — {reason}**", ""]
        out += ["```", (t.get("assistant") or "").strip(), "```"]
        return out

    lines += ["", "## Example kept traces (verbatim)"]
    for t in kept[:n_examples]:
        lines += _example(t, None)
    lines += ["", "## Example rejected traces (verbatim, with reasons)"]
    # show a spread of reasons, not three of the same kind
    seen: set[str] = set()
    shown = 0
    for t, reason in result["rejected"]:
        key = reason.split(":", 1)[0]
        if key in seen and shown < len(result["rejected"]) - 1:
            continue
        seen.add(key)
        lines += _example(t, reason)
        shown += 1
        if shown >= n_examples:
            break
    return "\n".join(lines) + "\n"
