"""Survivor-bias-proof SFT filter (training/filter.py) + pilot leakage guards.

Covers the four pinned rules from docs/training-risk-register.md risk 3 on
hand-built traces: a lucky-overconfident hit is rejected despite hitting, a
calibrated miss is kept, uncited numbers are rejected, the band-audit math is
exact, and no 2024/2025 data can enter the pilot.
"""
import json
from pathlib import Path

import pytest

from training.filter import (
    band_audit, band_of, brier_improvement, grounding_band, parse_final_p,
    run_filter, render_report, uncited_numbers, within_band,
)

ROOT = Path(__file__).resolve().parent.parent

USER_CTX = (
    "Candidate packet:\n"
    '{"player": "Test Player", "position": "WR", "adp_overall": 121.3, '
    '"adp_pos_rank": 45, "prior_season": {"games": 14, "total": 120.4, '
    '"ppg": 8.6, "pos_rank": 50}}\n'
    "Historical grounding:\n"
    '{"cohort_rate": 0.15, "cohort_n": 120, "position_base_rate": 0.12}\n'
)


def trace(assistant, outcome=0, anchor=0.15, tid="t0", user=USER_CTX):
    return {"trace_id": tid, "assistant": assistant, "user": user,
            "outcome": outcome, "anchor_rate": anchor, "anchor_n": 120,
            "season": 2019, "bench": "breakoutbench", "family": "full_slate",
            "player": "Test Player"}


def calibrated_text(p):
    return (f"The cohort rate of 15% anchors this. Prior season he was WR50 "
            f"on 8.6 ppg over 14 games — a modest profile, so a breakout "
            f"stays unlikely.\nFINAL PROBABILITY: {p}")


# ---------------------------------------------------------------------------
# parsing + primitives


def test_parse_final_p():
    assert parse_final_p("blah\nFINAL PROBABILITY: 0.23") == 0.23
    assert parse_final_p("Final Probability: 0.5") == 0.5  # case-insensitive
    assert parse_final_p("no final line, 0.4 maybe") is None
    assert parse_final_p("FINAL PROBABILITY: 1.7") is None  # out of range


def test_grounding_band():
    lo, hi = grounding_band(0.15)
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.75)
    assert grounding_band(0.30)[1] == pytest.approx(0.95)  # capped
    assert grounding_band(0.0)[0] == pytest.approx(0.02 / 3)  # floored anchor
    assert within_band(0.30, 0.15) and not within_band(0.90, 0.15)


def test_brier_improvement_scores_calibration_not_hits():
    # 30%-and-missed beats 90%-lucky-hit is NOT what raw Brier-vs-base says
    # (the lucky hit wins on its own question) — that is exactly why the
    # grounding gate (rule b) must run first. Here: improvement math itself.
    assert brier_improvement(0.15, 0.15, 0) == pytest.approx(0.0)
    # base 0.15, said 0.30, missed: worse than base on this question
    assert brier_improvement(0.30, 0.15, 0) == pytest.approx(0.0225 - 0.09)
    # base 0.15, said 0.10, missed: better than base
    assert brier_improvement(0.10, 0.15, 0) > 0


# ---------------------------------------------------------------------------
# rule (a): cite-or-don't-claim


def test_uncited_number_rejected():
    t = trace("He averaged 21.5 PPG last year and ran a 4.38 forty.\n"
              "FINAL PROBABILITY: 0.20")
    bad = uncited_numbers(t)
    assert "21.5" in bad and "4.38" in bad
    result = run_filter([t])
    assert not result["kept"]
    assert result["rejected"][0][1].startswith("uncited_number")


def test_cited_numbers_pass():
    t = trace(calibrated_text(0.15))
    assert uncited_numbers(t) == []


def test_percent_rescaling_and_small_int_exemption():
    # "15%" matches grounding 0.15; "3 of his last 4" is prose-safe
    t = trace("Cohort rate is 15%, and he was on the field for 3 of his "
              "last 4 games at 8.6 ppg.\nFINAL PROBABILITY: 0.18")
    assert uncited_numbers(t) == []


def test_final_probability_line_is_exempt():
    t = trace("Grounding says 0.15; I land slightly above.\n"
              "FINAL PROBABILITY: 0.22")
    assert uncited_numbers(t) == []


# ---------------------------------------------------------------------------
# rules (b) + (c): the survivor-bias core


def test_lucky_overconfident_trace_rejected_despite_hit():
    t = trace("Everything screams breakout. The 15% cohort rate is for other "
              "players.\nFINAL PROBABILITY: 0.90", outcome=1)  # it HIT
    result = run_filter([t])
    assert not result["kept"]
    reason = result["rejected"][0][1]
    assert reason.startswith("grounding_band")


def test_calibrated_miss_kept_over_worse_survivors():
    # same band, same outcome (both missed): the anchor-disciplined trace
    # survives the stratified Brier cut, the sloppier one does not
    traces = [
        trace(calibrated_text(0.12), outcome=0, tid="calibrated_miss"),
        trace(calibrated_text(0.19), outcome=0, tid="sloppy_miss"),
    ]
    result = run_filter(traces, keep_frac=0.6)
    kept_ids = {t["trace_id"] for t in result["kept"]}
    assert kept_ids == {"calibrated_miss"}  # a MISS is kept: quality, not hits
    reasons = {t["trace_id"]: r for t, r in result["rejected"]}
    assert reasons["sloppy_miss"].startswith("brier_cutoff")


def test_stratified_cut_preserves_band_hit_rates():
    # 8 gate-passing traces at p=0.65 (anchor 0.5), 4 hits + 4 misses.
    # A global Brier cut would keep hits only (survivorship); the stratified
    # cut must keep 2 hits + 2 misses, leaving the band hit rate unchanged.
    traces = [trace(calibrated_text(0.65), outcome=int(i < 4), anchor=0.5,
                    tid=f"s{i}") for i in range(8)]
    result = run_filter(traces, keep_frac=0.5)
    kept = result["kept"]
    assert len(kept) == 4
    assert sum(t["outcome"] for t in kept) == 2  # hit rate preserved at 0.5
    row = result["band_audit"][0]
    assert row["band"] == (0.6, 0.7)
    assert row["pre_hit_rate"] == row["kept_hit_rate"] == 0.5
    assert not row["flagged"]


def test_keep_decision_never_binary_outcome():
    # identical probability, opposite outcomes: gate decisions identical —
    # inclusion is never conditioned on the raw hit
    hit = trace(calibrated_text(0.90), outcome=1, tid="hit")
    miss = trace(calibrated_text(0.90), outcome=0, tid="miss")
    result = run_filter([hit, miss])
    reasons = {t["trace_id"]: r for t, r in result["rejected"]}
    assert reasons["hit"].split(":")[0] == reasons["miss"].split(":")[0] == \
        "grounding_band"


def test_unparseable_probability_rejected():
    result = run_filter([trace("Great player, buy everywhere.")])
    assert result["rejected"][0][1] == "unparseable_probability"


# ---------------------------------------------------------------------------
# rule (d): band-audit math


def test_band_of():
    assert band_of(0.0) == (0.0, 0.1)
    assert band_of(0.15) == (0.1, 0.2)
    assert band_of(1.0) == (0.9, 1.0)


def test_band_audit_math_exact():
    pre = [trace(calibrated_text(0.15), outcome=o, tid=f"a{i}")
           for i, o in enumerate([1, 0, 0, 1])]  # band [0.1,0.2): 2/4 hit
    for t in pre:
        t["p"] = 0.15
    kept = pre[:2]  # kept: outcomes 1, 0 -> 1/2 hit
    rows = band_audit(pre, kept)
    assert len(rows) == 1
    r = rows[0]
    assert r["band"] == (0.1, 0.2)
    assert r["n_pre"] == 4 and r["pre_hit_rate"] == 0.5
    assert r["n_kept"] == 2 and r["kept_hit_rate"] == 0.5
    assert r["diff"] == 0.0 and not r["flagged"]


def test_band_audit_flags_survivorship_jump():
    # 20 traces at p~0.15, 4 hits (20%); keep only the 4 hits -> kept hit
    # rate 100%, a blatant survivorship signature that must flag
    pre = [trace(calibrated_text(0.15), outcome=int(i < 4), tid=f"b{i}")
           for i in range(20)]
    for t in pre:
        t["p"] = 0.15
    kept = pre[:5]  # 4 hits + 1 miss = 80% vs 20% pre
    rows = band_audit(pre, kept)
    assert rows[0]["flagged"]


def test_report_renders():
    result = run_filter([trace(calibrated_text(0.15), tid="k"),
                         trace("uncited 99.9 stat\nFINAL PROBABILITY: 0.15",
                               tid="r")])
    md = render_report(result)
    assert "Survivorship audit" in md and "uncited_number" in md
    assert "FINAL PROBABILITY" in md  # verbatim examples included


# ---------------------------------------------------------------------------
# leakage: no 2024/2025 in the pilot


def test_pilot_season_constants():
    from training import sft_datagen as dg
    assert dg.SFT_VAL_SEASON == 2024 and dg.HOLDOUT_SEASON == 2025
    assert dg.SFT_VAL_SEASON not in dg.PILOT_SEASONS
    assert dg.HOLDOUT_SEASON not in dg.PILOT_SEASONS
    assert max(dg.PILOT_SEASONS) == 2023 and min(dg.PILOT_SEASONS) == 2016


def test_no_2024_2025_leakage_in_pilot_outputs():
    for name in ("raw_traces_v0.jsonl", "pilot_v0.jsonl"):
        path = ROOT / "data" / "processed" / "sft" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            season = rec.get("season") or rec["meta"]["season"]
            assert 2016 <= season <= 2023, f"leakage in {name}: {season}"
            as_of = rec.get("as_of") or rec["meta"]["as_of"]
            assert as_of == f"{season}-09-01"


def test_pilot_sample_is_deterministic_and_in_bounds():
    from training.sft_datagen import build_pilot_sample
    s1, s2 = build_pilot_sample(), build_pilot_sample()
    assert s1 == s2, "pilot sample must be deterministic"
    assert {it["season"] for it in s1} <= set(range(2016, 2024))
    assert all(it["outcome"] in (0, 1) for it in s1)
    assert all(it["player"] for it in s1)  # named track
    fams = {it["family"] for it in s1}
    assert fams == {"full_slate", "bust", "season_threshold"}
    assert 140 <= len(s1) <= 160
