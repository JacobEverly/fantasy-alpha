"""Canary protocol v1 guards: swap correctness, KEY audit trail, no outcome
leakage, determinism, arm construction, index math, and the checkpoint gate."""
import json

import pytest

from evals.breakoutbench import ADP_DIR, LABELS, QUESTIONS_DIR
from evals.canaries import (
    GATE_INDEX_CI_LOW,
    PRESET,
    SWAP_TARGET_FRAC,
    _n_swap,
    _pair_credit,
    bootstrap_index,
    build_canary_set,
    canary_gate,
    load_answers_files,
    main,
    score_canaries,
    season_pair_stats,
)

HAVE_DATA = ADP_DIR.exists() and (LABELS / "breakouts.csv").exists() and \
    (QUESTIONS_DIR / f"breakoutbench_2019_{PRESET}.json").exists()
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="requires data/ + built question sets")


# ---------------------------------------------------------------------------
# hand-built: pair credit, index math, gate


def test_pair_credit():
    assert _pair_credit(0.7, 0.2) == 1.0
    assert _pair_credit(0.2, 0.7) == 0.0
    assert _pair_credit(0.5, 0.5) == 0.5  # tie
    assert _pair_credit(None, 0.5) is None  # unanswered pair drops out
    assert _pair_credit(0.5, None) is None


def _mini_key():
    """2 control pairs + 2 swapped pairs. Ids C1a/C1b etc; outcome-1 member
    listed first in each pair. In swapped pairs the breakout packet sits
    under the NON-breakout id (packet_outcome inverted vs outcome)."""
    q = {}
    for pid, (a, b) in enumerate((("C1a", "C1b"), ("C2a", "C2b"))):
        q[a] = {"arm": "control", "pair_id": pid, "player": a, "outcome": 1,
                "packet_owner": a, "packet_outcome": 1}
        q[b] = {"arm": "control", "pair_id": pid, "player": b, "outcome": 0,
                "packet_owner": b, "packet_outcome": 0}
    for pid, (a, b) in enumerate((("S1a", "S1b"), ("S2a", "S2b")), start=2):
        q[a] = {"arm": "swapped", "pair_id": pid, "player": a, "outcome": 1,
                "packet_owner": b, "packet_outcome": 0}
        q[b] = {"arm": "swapped", "pair_id": pid, "player": b, "outcome": 0,
                "packet_owner": a, "packet_outcome": 1}
    q["U0"] = {"arm": "unpaired", "pair_id": None, "player": "U0", "outcome": 0,
               "packet_owner": "U0", "packet_outcome": 0}
    return {"questions": q}


def test_season_pair_stats_hand_built():
    answers = {
        "C1a": 0.8, "C1b": 0.2,   # control correct
        "C2a": 0.1, "C2b": 0.6,   # control wrong
        "S1a": 0.2, "S1b": 0.9,   # follows the packet (breakout packet on S1b)
        "S2a": 0.9, "S2b": 0.1,   # follows the id (identity leak)
        "U0": 0.5,                # unpaired: ignored
    }
    st = season_pair_stats(_mini_key(), answers)
    assert st["control"] == [1.0, 0.0]
    assert st["swap_pf"] == [1.0, 0.0]


def test_index_math_and_gate_directions():
    # identity-follower: perfect on control, anti-follows packets when swapped
    leak = {s: season_pair_stats(_mini_key(), {
        "C1a": 0.9, "C1b": 0.1, "C2a": 0.9, "C2b": 0.1,
        "S1a": 0.9, "S1b": 0.1, "S2a": 0.9, "S2b": 0.1})
        for s in (2015, 2016, 2021, 2022)}
    from evals.canaries import _pool_stats
    pooled = _pool_stats(leak, sorted(leak))
    assert pooled["a_control"] == 1.0 and pooled["a_swap_pf"] == 0.0
    assert pooled["index"] == 1.0
    ci = bootstrap_index(leak, reps=200, seed=7)
    assert ci["index"] == (1.0, 1.0)
    assert canary_gate({"pooled": pooled, "ci": ci})["pass"] is False

    # packet-follower: identical accuracy on both arms -> index 0, gate PASS
    clean = {s: season_pair_stats(_mini_key(), {
        "C1a": 0.9, "C1b": 0.1, "C2a": 0.9, "C2b": 0.1,
        "S1a": 0.1, "S1b": 0.9, "S2a": 0.1, "S2b": 0.9})
        for s in (2015, 2016)}
    pooled = _pool_stats(clean, sorted(clean))
    assert pooled["index"] == 0.0
    ci = bootstrap_index(clean, reps=200, seed=7)
    assert canary_gate({"pooled": pooled, "ci": ci})["pass"] is True


def test_gate_requires_scored_pairs():
    g = canary_gate({"pooled": {"index": None}, "ci": {}})
    assert g["pass"] is False and "cannot run" in g["reason"]


def test_bootstrap_deterministic():
    per_season = {s: season_pair_stats(_mini_key(), {
        "C1a": 0.8, "C1b": 0.2, "C2a": 0.1 * (s % 3), "C2b": 0.6,
        "S1a": 0.2, "S1b": 0.9, "S2a": 0.9 * (s % 2), "S2b": 0.1})
        for s in (2015, 2016, 2020, 2021)}
    a = bootstrap_index(per_season, reps=500, seed=11)
    b = bootstrap_index(per_season, reps=500, seed=11)
    assert a == b  # same seed -> byte-identical CIs
    from evals.canaries import _pool_stats
    pooled = _pool_stats(per_season, sorted(per_season))
    lo, hi = a["index"]
    assert lo <= pooled["index"] <= hi  # CI brackets the point estimate


def test_n_swap_bounds():
    assert _n_swap(0, 50) == 0
    assert _n_swap(1, 50) == 1
    # >=2 pairs always leaves >=1 control pair
    for n_pairs in range(2, 20):
        n = _n_swap(n_pairs, 60)
        assert 1 <= n <= n_pairs - 1
    # the 30%-of-candidates cap binds when the slate is small
    assert _n_swap(10, 20) * 2 <= max(SWAP_TARGET_FRAC * 20, 2)


# ---------------------------------------------------------------------------
# real question sets: swap correctness, audit trail, leakage, determinism


@needs_data
def test_swap_correctness_and_key_audit():
    qpayload, kpayload = build_canary_set(2019, PRESET)
    orig = json.loads((QUESTIONS_DIR / f"breakoutbench_2019_{PRESET}.json").read_text())
    orig_packets = {q["question_id"]: q["packet"] for q in orig["questions"]
                    if q["family"] == "full_slate"}
    okey = json.loads(
        (QUESTIONS_DIR / f"breakoutbench_2019_{PRESET}_KEY.json").read_text())["questions"]

    # ids stable: exactly the original full_slate ids, no additions
    canary_packets = {q["question_id"]: q["packet"] for q in qpayload["questions"]}
    assert set(canary_packets) == set(orig_packets)

    by_pair = {}
    for qid, info in kpayload["questions"].items():
        # own player/outcome always match the original benchmark KEY
        assert info["player"] == okey[qid]["player"]
        assert info["outcome"] == okey[qid]["outcome"]
        if info["arm"] == "swapped":
            by_pair.setdefault(info["pair_id"], []).append(qid)
        else:
            # unswapped questions: packet untouched, owner is self
            assert canary_packets[qid] == orig_packets[qid]
            assert info["packet_owner"] == info["player"]
            assert info["packet_outcome"] == info["outcome"]

    assert len(by_pair) == kpayload["n_swapped_pairs"] > 0
    for pair_id, (qa, qb) in ((k, tuple(v)) for k, v in by_pair.items()):
        ia, ib = kpayload["questions"][qa], kpayload["questions"][qb]
        # packets exchanged exactly
        assert canary_packets[qa] == orig_packets[qb]
        assert canary_packets[qb] == orig_packets[qa]
        # audit trail is a consistent involution
        assert ia["packet_owner"] == ib["player"]
        assert ib["packet_owner"] == ia["player"]
        assert ia["packet_outcome"] == ib["outcome"]
        assert ib["packet_outcome"] == ia["outcome"]
        # pairs: same position, different outcomes
        assert ia["outcome"] != ib["outcome"]
        assert canary_packets[qa]["position"] == canary_packets[qb]["position"]


@needs_data
def test_pairs_are_adp_adjacent_same_position():
    _, kpayload = build_canary_set(2016, PRESET)
    orig = json.loads((QUESTIONS_DIR / f"breakoutbench_2016_{PRESET}.json").read_text())
    packets = {q["question_id"]: q["packet"] for q in orig["questions"]
               if q["family"] == "full_slate"}
    pairs = {}
    for qid, info in kpayload["questions"].items():
        if info["pair_id"] is not None:
            pairs.setdefault(info["pair_id"], []).append(qid)
    assert pairs
    for qa, qb in pairs.values():
        assert packets[qa]["position"] == packets[qb]["position"]


@needs_data
def test_no_outcome_or_name_leakage_in_question_payload():
    qpayload, kpayload = build_canary_set(2019, PRESET)
    blob = json.dumps(qpayload)
    for forbidden in ("outcome", "breakout\"", "player\"", "packet_owner",
                      "finish_pos_rank", "arm\"", "pair_id"):
        assert forbidden not in blob
    for info in kpayload["questions"].values():  # no real names anywhere
        assert info["player"] not in blob
        assert info["packet_owner"] not in blob


@needs_data
def test_build_is_deterministic():
    assert build_canary_set(2018, PRESET) == build_canary_set(2018, PRESET)


@needs_data
def test_swapped_candidate_fraction_capped():
    for season in (2015, 2016, 2019, 2022, 2024):
        _, k = build_canary_set(season, PRESET)
        assert k["swapped_candidate_frac"] <= SWAP_TARGET_FRAC + 1e-9
        assert 1 <= k["n_swapped_pairs"] <= k["n_pairs"]
        if k["n_pairs"] >= 2:  # both arms populated
            assert k["n_swapped_pairs"] < k["n_pairs"]


@needs_data
def test_holdout_refused():
    with pytest.raises(SystemExit):
        main(["build", "--season", "2025"])


@needs_data
def test_score_end_to_end_from_answers_files(tmp_path):
    """Synthetic packet-follower answers over the real 2019+2021 canary keys
    -> index 0, gate PASS; and load_answers_files drops junk entries."""
    paths = []
    for season in (2019, 2021):
        key = json.loads(
            (QUESTIONS_DIR / f"canary_{season}_{PRESET}_KEY.json").read_text())
        answers = [{"question_id": qid, "p": 0.9 if info["packet_outcome"] else 0.1}
                   for qid, info in key["questions"].items()]
        answers += [{"question_id": "NOPE", "p": 0.5},
                    {"question_id": next(iter(key["questions"])), "p": 7.0}]
        p = tmp_path / f"model_{season}_{PRESET}.json"
        p.write_text(json.dumps(answers))
        paths.append(p)
    result = score_canaries(load_answers_files(paths), reps=300, seed=3)
    assert result["pooled"]["a_control"] == 1.0
    assert result["pooled"]["a_swap_pf"] == 1.0
    assert result["pooled"]["index"] == 0.0
    assert result["gate"]["pass"] is True
    assert "2015-2019" in result["eras"] and "2020-2024" in result["eras"]
