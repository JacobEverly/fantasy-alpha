#!/usr/bin/env python3
"""Memorization canaries — canary protocol v1 (contamination detector).

Implements docs/breakoutbench-design.md §2's memorization canary for the
anonymized track: counterfactually perturbed dossiers whose scoring separates
"follows the features" from "follows the hidden identity".

Protocol v1 (frozen with this module; changes bump the version):

1. SWAP CANARIES. Within each ppr full_slate question set (2015-2024), pair
   candidates at the same position with DIFFERENT outcomes (one breakout, one
   not), matched to be ADP-adjacent (nearest adp_pos_rank) so the within-pair
   market signal is minimal. A seeded coin assigns pairs to two arms:
   - swapped arm (capped so swapped candidates stay <= ~30% of the slate):
     the two candidates' ENTIRE feature packets are exchanged. Each keeps its
     anon question_id; the sequestered canary KEY records the swap (who owns
     the id, whose packet it now carries, both outcomes, the pair id).
   - control arm: packets untouched.

2. METRICS (pairwise, comparative):
   - A_control  = share of control pairs where the higher p lands on the
     candidate that historically broke out (own coherent packet).
   - A_swap_pf  = packet-following accuracy on swapped pairs: share where the
     higher p lands on the QUESTION CARRYING the breakout packet (which now
     sits under the other candidate's id).
   - canary sensitivity index = A_control - A_swap_pf, bootstrap 95% CI by
     resampling SEASONS (the independent unit), reported pooled and per era
     (2015-2019 vs 2020-2024).
   Ties in p score 0.5. A model whose probabilities are a pure function of
   packet CONTENT follows the packet wherever it moves -> index ~ 0. A model
   that has attached outcomes to the hidden identities behind the anon ids
   (the failure mode when eval questions/keys leak into training data) keeps
   betting on the id after its packet moves -> A_swap_pf collapses, index >> 0.

   Honest scope note: a full-packet swap moves the profile fingerprint WITH
   the packet, so pretraining recall keyed on the fingerprint itself (the
   suspected source of the naked-Qwen era anomaly) is swap-INVARIANT and does
   not move this index. Protocol v1 therefore carries a second probe:

3. FINGERPRINT PROBE (no new inference — reuses existing answers). On the
   UNSWAPPED originals, within-pair discrimination accuracy per era for the
   model vs. a pure-code feature reference (the no-peek cohort/prior-rank
   grounding rates of evals/run_expanded_qwen.py). Pairs are ADP-adjacent, so
   legitimate feature headroom is small and era-stable; model accuracy that
   materially exceeds the reference ONLY in 2020-2024 is evidence that the
   era anomaly rides on profile-fingerprint recall inside the packets.

4. CANARY GATE (standard for every future checkpoint): a checkpoint FAILS
   the gate iff the pooled canary sensitivity index 95% CI lower bound
   exceeds GATE_INDEX_CI_LOW (0.05). A failing checkpoint cannot claim its
   anonymized-track numbers (they are id-recall, not reasoning); the gate
   line must appear next to any published BreakoutBench score. Wire-up: run
   `canaries.py run` alongside the BreakoutBench battery for each checkpoint
   and paste the gate verdict into the eval report.

CLI:
  build  -> evals/questions/canary_<season>_ppr.json (+ _KEY.json, sequestered)
  score  -> index / arms / eras / gate from answers files
  run    -> naked Qwen/Qwen3.5-9B over the canary sets (temp 0, budget cap
            $1.00, ledger to docs/budget-ledger.md), then score + fingerprint
            probe -> evals/results/canaries.md
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evals.breakoutbench import (  # noqa: E402
    ANSWERS_DIR,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    ERAS,
    HOLDOUT_SEASON,
    INSTRUCTIONS,
    QUESTIONS_DIR,
    SEASONS,
    season_formats,
)

CANARY_VERSION = "canary_v1"
PRESET = "ppr"  # protocol v1 runs the ppr grid (the era anomaly's home turf)
SWAP_TARGET_FRAC = 0.30  # cap: swapped candidates <= ~30% of the full_slate set
PAIR_SEED = 20260808
GATE_INDEX_CI_LOW = 0.05  # gate fails iff pooled index CI lower bound > this

MODEL = "Qwen/Qwen3.5-9B"
BUDGET_CAP = 1.00
MAX_REQUEST_COST = 0.02

RESULTS_MD = ROOT / "evals" / "results" / "canaries.md"


# ---------------------------------------------------------------------------
# build


def _pair_candidates(fs_questions: list[dict], key: Mapping[str, dict]) -> list[tuple[dict, dict]]:
    """Disjoint same-position (breakout, non-breakout) pairs, ADP-adjacent.

    Deterministic: positions in sorted order, breakouts walked in
    adp_pos_rank order, each greedily matched to the nearest unused
    non-breakout by |delta adp_pos_rank| (question_id breaks ties)."""
    by_pos: dict[str, list[dict]] = defaultdict(list)
    for q in fs_questions:
        by_pos[q["packet"]["position"]].append(q)
    pairs: list[tuple[dict, dict]] = []
    for pos in sorted(by_pos):
        cands = sorted(by_pos[pos],
                       key=lambda q: (q["packet"]["adp_pos_rank"], q["question_id"]))
        breakouts = [q for q in cands if key[q["question_id"]]["outcome"] == 1]
        others = [q for q in cands if key[q["question_id"]]["outcome"] == 0]
        used: set[str] = set()
        for b in breakouts:
            best = None
            for c in others:
                if c["question_id"] in used:
                    continue
                d = abs(c["packet"]["adp_pos_rank"] - b["packet"]["adp_pos_rank"])
                cand = (d, c["question_id"], c)
                if best is None or cand[:2] < best[:2]:
                    best = cand
            if best is not None:
                used.add(best[1])
                pairs.append((b, best[2]))
    pairs.sort(key=lambda p: p[0]["question_id"])
    return pairs


def _n_swap(n_pairs: int, n_candidates: int) -> int:
    """Swapped-arm size: about half the pairs, capped so swapped candidates
    stay <= SWAP_TARGET_FRAC of the set; always leaves >=1 control pair when
    there are >=2 pairs."""
    if n_pairs == 0:
        return 0
    cap = int(SWAP_TARGET_FRAC * n_candidates) // 2
    n = min((n_pairs + 1) // 2, max(cap, 1))
    if n_pairs >= 2:
        n = min(n, n_pairs - 1)
    return max(n, 1)


def build_canary_set(season: int, preset: str = PRESET,
                     questions_dir: Path = QUESTIONS_DIR) -> tuple[dict, dict]:
    """(canary_question_payload, canary_key_payload) for one season-format.

    Reads the frozen BreakoutBench question set + KEY; never touches labels
    directly. Question ids are the original anon ids (stable); outcomes and
    names live only in the canary KEY."""
    assert season != HOLDOUT_SEASON, "2025 is the untouched holdout"
    qpayload = json.loads(
        (questions_dir / f"breakoutbench_{season}_{preset}.json").read_text())
    key = json.loads(
        (questions_dir / f"breakoutbench_{season}_{preset}_KEY.json").read_text())["questions"]
    fs = [json.loads(json.dumps(q)) for q in qpayload["questions"]
          if q["family"] == "full_slate"]

    pairs = _pair_candidates(fs, key)
    rng = random.Random(f"{CANARY_VERSION}:{season}:{preset}:{PAIR_SEED}")
    order = list(range(len(pairs)))
    rng.shuffle(order)
    n_swap = _n_swap(len(pairs), len(fs))
    swapped_idx = set(order[:n_swap])

    arm_of: dict[str, tuple[str, int]] = {}  # qid -> (arm, pair_id)
    by_id = {q["question_id"]: q for q in fs}
    for pair_id, (b, c) in enumerate(pairs):
        arm = "swapped" if pair_id in swapped_idx else "control"
        arm_of[b["question_id"]] = (arm, pair_id)
        arm_of[c["question_id"]] = (arm, pair_id)
        if arm == "swapped":
            bq, cq = by_id[b["question_id"]], by_id[c["question_id"]]
            bq["packet"], cq["packet"] = cq["packet"], bq["packet"]

    key_questions = {}
    for q in fs:
        qid = q["question_id"]
        arm, pair_id = arm_of.get(qid, ("unpaired", None))
        own = key[qid]
        if arm == "swapped":
            b, c = pairs[pair_id]
            donor_qid = c["question_id"] if qid == b["question_id"] else b["question_id"]
            donor = key[donor_qid]
        else:
            donor = own
        key_questions[qid] = {
            "arm": arm,
            "pair_id": pair_id,
            "player": own["player"],
            "outcome": own["outcome"],
            "packet_owner": donor["player"],
            "packet_outcome": donor["outcome"],
        }

    fs.sort(key=lambda q: q["question_id"])
    n_swapped_cands = 2 * n_swap
    question_payload = {
        "benchmark": f"breakoutbench_{CANARY_VERSION}",
        "variant": "anonymized_canary",
        "protocol": CANARY_VERSION,
        "season_masked": True,
        "format": preset,
        "families": {"full_slate": len(fs)},
        "instructions": INSTRUCTIONS,
        "questions": fs,
    }
    key_payload = {
        "benchmark": f"breakoutbench_{CANARY_VERSION}",
        "season": season,
        "format": preset,
        "n_pairs": len(pairs),
        "n_swapped_pairs": n_swap,
        "swapped_candidate_frac": round(n_swapped_cands / len(fs), 4) if fs else 0.0,
        "questions": key_questions,
    }
    return question_payload, key_payload


def build(seasons: Sequence[int] = SEASONS, preset: str = PRESET,
          out_dir: Path = QUESTIONS_DIR,
          questions_dir: Path = QUESTIONS_DIR) -> list[tuple[Path, Path]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for season, p in season_formats(seasons):
        if p != preset:
            continue
        qpayload, kpayload = build_canary_set(season, preset, questions_dir)
        qpath = out_dir / f"canary_{season}_{preset}.json"
        kpath = out_dir / f"canary_{season}_{preset}_KEY.json"
        qpath.write_text(json.dumps(qpayload, indent=1) + "\n")
        kpath.write_text(json.dumps(kpayload, indent=1) + "\n")
        written.append((qpath, kpath))
        print(f"built canary set {season}/{preset}: {kpayload['n_pairs']} pairs "
              f"({kpayload['n_swapped_pairs']} swapped, "
              f"{kpayload['swapped_candidate_frac']:.0%} of candidates) -> {qpath.name}")
    return written


# ---------------------------------------------------------------------------
# scoring


def _pair_credit(p_hi: float | None, p_lo: float | None) -> float | None:
    """1 if p_hi > p_lo, 0.5 on tie, 0 otherwise; None if either is missing
    (unanswered pairs drop out rather than polluting the accuracy)."""
    if p_hi is None or p_lo is None:
        return None
    if p_hi > p_lo:
        return 1.0
    if p_hi < p_lo:
        return 0.0
    return 0.5


def season_pair_stats(key_payload: Mapping, answers: Mapping[str, float]) -> dict:
    """{'control': [credit...], 'swap_pf': [credit...]} for one season.

    control: credit for putting the higher p on the true-breakout candidate.
    swap_pf: packet-following credit — higher p on the question CARRYING the
    breakout packet (which sits under the non-breakout candidate's id)."""
    by_pair: dict[int, list[tuple[str, dict]]] = defaultdict(list)
    for qid, info in key_payload["questions"].items():
        if info["arm"] in ("control", "swapped"):
            by_pair[info["pair_id"]].append((qid, info))
    out: dict[str, list[float]] = {"control": [], "swap_pf": []}
    for pair_id in sorted(by_pair):
        members = by_pair[pair_id]
        assert len(members) == 2, f"pair {pair_id} has {len(members)} members"
        (qa, ia), (qb, ib) = members
        assert ia["outcome"] != ib["outcome"], f"pair {pair_id} outcomes equal"
        arm = ia["arm"]
        if arm == "control":
            hi = qa if ia["outcome"] == 1 else qb
        else:  # swapped: follow the packet, not the id
            hi = qa if ia["packet_outcome"] == 1 else qb
        lo = qb if hi == qa else qa
        credit = _pair_credit(answers.get(hi), answers.get(lo))
        if credit is not None:
            out["control" if arm == "control" else "swap_pf"].append(credit)
    return out


def _accuracy(credits: Sequence[float]) -> float | None:
    return sum(credits) / len(credits) if credits else None


def _pool_stats(per_season: Mapping[int, dict], seasons: Sequence[int]) -> dict:
    """Pool pair credits over a multiset of seasons (bootstrap draws repeat)."""
    control: list[float] = []
    swap_pf: list[float] = []
    for s in seasons:
        st = per_season.get(s)
        if st:
            control.extend(st["control"])
            swap_pf.extend(st["swap_pf"])
    a_c, a_s = _accuracy(control), _accuracy(swap_pf)
    return {
        "a_control": a_c, "n_control": len(control),
        "a_swap_pf": a_s, "n_swap": len(swap_pf),
        "index": (a_c - a_s) if (a_c is not None and a_s is not None) else None,
    }


def bootstrap_index(per_season: Mapping[int, dict], reps: int = BOOTSTRAP_REPS,
                    seed: int = BOOTSTRAP_SEED) -> dict:
    """95% CIs for a_control / a_swap_pf / index by resampling seasons."""
    seasons = sorted(per_season)
    rng = random.Random(seed)
    stats: dict[str, list[float]] = defaultdict(list)
    for _ in range(reps):
        draw = [seasons[rng.randrange(len(seasons))] for _ in seasons]
        pooled = _pool_stats(per_season, draw)
        for k in ("a_control", "a_swap_pf", "index"):
            if pooled[k] is not None:
                stats[k].append(pooled[k])

    def ci(values: list[float]) -> tuple[float, float] | None:
        if not values:
            return None
        vs = sorted(values)
        return (vs[int(0.025 * (len(vs) - 1))], vs[int(0.975 * (len(vs) - 1))])

    return {k: ci(v) for k, v in stats.items()}


def score_canaries(answers_by_season: Mapping[int, Mapping[str, float]],
                   questions_dir: Path = QUESTIONS_DIR, preset: str = PRESET,
                   reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    per_season = {}
    for season, answers in answers_by_season.items():
        kpath = questions_dir / f"canary_{season}_{preset}_KEY.json"
        per_season[season] = season_pair_stats(json.loads(kpath.read_text()), answers)
    seasons = sorted(per_season)
    result = {
        "seasons": seasons,
        "pooled": _pool_stats(per_season, seasons),
        "ci": bootstrap_index(per_season, reps, seed),
        "eras": {},
        "per_season": {s: {"control": _accuracy(st["control"]),
                           "n_control": len(st["control"]),
                           "swap_pf": _accuracy(st["swap_pf"]),
                           "n_swap": len(st["swap_pf"])}
                       for s, st in per_season.items()},
    }
    for era_name, era_seasons in ERAS:
        in_era = [s for s in seasons if s in era_seasons]
        if not in_era:
            continue
        sub = {s: per_season[s] for s in in_era}
        result["eras"][era_name] = {
            "pooled": _pool_stats(sub, in_era),
            "ci": bootstrap_index(sub, reps, seed),
        }
    result["gate"] = canary_gate(result)
    return result


def canary_gate(result: Mapping) -> dict:
    """The standard checkpoint gate. FAIL iff the pooled index 95% CI lower
    bound exceeds GATE_INDEX_CI_LOW — the model's swapped-pair accuracy
    collapsed relative to control, i.e. it followed the hidden identity, not
    the packet. A failing checkpoint's anonymized-track numbers are void."""
    ci = result["ci"].get("index")
    idx = result["pooled"]["index"]
    if ci is None or idx is None:
        return {"pass": False, "reason": "no scored pairs — gate cannot run"}
    failed = ci[0] > GATE_INDEX_CI_LOW
    return {
        "pass": not failed,
        "index": idx,
        "ci": list(ci),
        "threshold_ci_low": GATE_INDEX_CI_LOW,
        "reason": (f"index {idx:.3f} CI [{ci[0]:.3f}, {ci[1]:.3f}] "
                   + ("resolves above" if failed else "does not resolve above")
                   + f" {GATE_INDEX_CI_LOW}"),
    }


def load_answers_files(paths: Sequence[Path]) -> dict[int, dict[str, float]]:
    """answers files (named *_<season>_<preset>*.json) -> {season: {qid: p}}.
    First answer per id wins; out-of-range p is dropped."""
    from evals.breakoutbench import season_preset_of
    out: dict[int, dict[str, float]] = defaultdict(dict)
    for path in paths:
        season, _preset = season_preset_of(Path(path))
        for a in json.loads(Path(path).read_text()):
            qid = str(a["question_id"])
            try:
                p = float(a["p"])
            except (TypeError, ValueError):
                continue
            if qid not in out[season] and 0.0 <= p <= 1.0:
                out[season][qid] = p
    return dict(out)


# ---------------------------------------------------------------------------
# fingerprint probe (secondary — no new inference)


def fingerprint_probe(preset: str = PRESET,
                      questions_dir: Path = QUESTIONS_DIR,
                      answers_dir: Path = ANSWERS_DIR,
                      run_name: str = "naked") -> dict | None:
    """Within-pair accuracy on the UNSWAPPED originals, per era: existing
    model answers vs the pure-code no-peek feature reference. Pairs are the
    canary pairs (both arms — packets are coherent in the original files).
    Returns None when the original answers are not on disk."""
    from evals.run_expanded_qwen import _family_rows, _prior_ranks, grounding_for

    fam_rows = _family_rows(preset)
    prior_ranks = _prior_ranks(preset)
    per_season_model: dict[int, dict] = {}
    per_season_ref: dict[int, dict] = {}
    for season, p in season_formats(SEASONS):
        if p != preset:
            continue
        apath = answers_dir / f"qwen_{run_name}_{season}_{preset}_full_slate.json"
        kpath = questions_dir / f"canary_{season}_{preset}_KEY.json"
        if not apath.exists() or not kpath.exists():
            continue
        model_p = {str(a["question_id"]): float(a["p"])
                   for a in json.loads(apath.read_text())}
        orig = json.loads(
            (questions_dir / f"breakoutbench_{season}_{preset}.json").read_text())
        packets = {q["question_id"]: q["packet"] for q in orig["questions"]
                   if q["family"] == "full_slate"}
        past = [r for r in fam_rows.get("full_slate", []) if r["season"] < season]

        def ref_p(qid: str) -> float | None:
            g = grounding_for("full_slate", packets[qid], past, prior_ranks, season)
            for k in ("prior_rank_cohort_rate", "cohort_rate", "position_base_rate"):
                if g.get(k) is not None:
                    return g[k]
            return None

        key_payload = json.loads(kpath.read_text())
        pairs: dict[int, list[tuple[str, dict]]] = defaultdict(list)
        for qid, info in key_payload["questions"].items():
            if info["arm"] in ("control", "swapped"):
                pairs[info["pair_id"]].append((qid, info))
        m_credits, r_credits = [], []
        for pair_id in sorted(pairs):
            (qa, ia), (qb, ib) = pairs[pair_id]
            hi, lo = (qa, qb) if ia["outcome"] == 1 else (qb, qa)
            c = _pair_credit(model_p.get(hi), model_p.get(lo))
            if c is not None:
                m_credits.append(c)
            c = _pair_credit(ref_p(hi), ref_p(lo))
            if c is not None:
                r_credits.append(c)
        per_season_model[season] = {"control": m_credits, "swap_pf": []}
        per_season_ref[season] = {"control": r_credits, "swap_pf": []}

    if not per_season_model:
        return None
    out = {"run": run_name, "eras": {}}
    for era_name, era_seasons in ERAS:
        in_era = [s for s in per_season_model if s in era_seasons]
        if not in_era:
            continue
        sub_m = {s: per_season_model[s] for s in in_era}
        sub_r = {s: per_season_ref[s] for s in in_era}
        out["eras"][era_name] = {
            "model": _pool_stats(sub_m, in_era),
            "model_ci": bootstrap_index(sub_m, BOOTSTRAP_REPS, BOOTSTRAP_SEED),
            "reference": _pool_stats(sub_r, in_era),
            "reference_ci": bootstrap_index(sub_r, BOOTSTRAP_REPS, BOOTSTRAP_SEED),
        }
    return out


# ---------------------------------------------------------------------------
# run (naked Qwen over the canary sets)


def run_model(seasons: Sequence[int] = SEASONS, preset: str = PRESET,
              questions_dir: Path = QUESTIONS_DIR,
              answers_dir: Path = ANSWERS_DIR) -> dict:
    """One request per canary season set; resume-safe; budget-capped."""
    from evals.run_expanded_qwen import (PRICE_IN, PRICE_OUT, build_prompt,
                                         chat, parse_answers)
    answers_dir.mkdir(parents=True, exist_ok=True)
    spend = {"in": 0, "out": 0, "cost": 0.0, "requests": 0}
    failures: list[dict] = []
    for season, p in season_formats(seasons):
        if p != preset:
            continue
        out_path = answers_dir / f"qwen_canary_{season}_{preset}.json"
        if out_path.exists():
            continue
        if spend["cost"] + MAX_REQUEST_COST > BUDGET_CAP:
            raise SystemExit(f"BUDGET CAP: ${spend['cost']:.3f} spent, next request "
                             f"could cross ${BUDGET_CAP:.2f} — aborting")
        qpayload = json.loads(
            (questions_dir / f"canary_{season}_{preset}.json").read_text())
        messages, valid_ids = build_prompt(qpayload, "full_slate", grounded=False)
        try:
            text, usage = chat(messages)
            try:
                answers = parse_answers(text, valid_ids)
            except Exception as e:  # one repair retry
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Invalid ({e}). Return ONLY the JSON "
                              "array, one object per candidate, covering every question_id."}]
                text, usage2 = chat(messages)
                usage = {k: usage.get(k, 0) + usage2.get(k, 0) for k in set(usage) | set(usage2)}
                answers = parse_answers(text, valid_ids)
        except Exception as e:
            failures.append({"season": season, "format": preset, "error": str(e)})
            print(f"  FAIL canary {season} {preset}: {e}", flush=True)
            continue
        spend["in"] += usage.get("prompt_tokens", 0)
        spend["out"] += usage.get("completion_tokens", 0)
        spend["cost"] = spend["in"] * PRICE_IN + spend["out"] * PRICE_OUT
        spend["requests"] += 1
        out_path.write_text(json.dumps(answers, indent=1) + "\n")
        print(f"  canary {season} {preset}: {len(answers)}/{len(valid_ids)} answered "
              f"({usage.get('prompt_tokens')}in/{usage.get('completion_tokens')}out, "
              f"total ${spend['cost']:.3f})", flush=True)
    spend["failures"] = failures
    return spend


# ---------------------------------------------------------------------------
# report


def _fmt_ci(ci, digits: int = 3) -> str:
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]" if ci else "—"


def _fmt(v, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if v is not None else "—"


def verdict_lines(result: Mapping, probe: Mapping | None) -> list[str]:
    """Data-driven verdict on the hypothesis that the naked-model era anomaly
    (2020-2024 lift >> 2015-2019) is memorization leaking through
    anonymization. Evidence read: (1) does the swap index resolve positive in
    2020-2024 (identity-channel leak)? (2) does the fingerprint probe show
    model-over-reference within-pair excess concentrated in 2020-2024?"""
    lines = []
    recent = result["eras"].get("2020-2024")
    early = result["eras"].get("2015-2019")
    if recent and recent["ci"].get("index"):
        lo, hi = recent["ci"]["index"]
        idx = recent["pooled"]["index"]
        if lo > GATE_INDEX_CI_LOW:
            lines.append(f"- 2020-2024 swap index {idx:.3f} [{lo:.3f}, {hi:.3f}] resolves "
                         "positive: identity-channel leakage in the recent era — "
                         "**consistent with the memorization hypothesis**.")
        else:
            lines.append(f"- 2020-2024 swap index {idx:.3f} [{lo:.3f}, {hi:.3f}] does NOT "
                         "resolve above the gate threshold: no detectable identity-channel "
                         "leak (and none is expected pre-training-contamination — see scope "
                         "note).")
    if early and recent and early["ci"].get("index") and recent["ci"].get("index"):
        e_lo, e_hi = early["ci"]["index"]
        r_lo, r_hi = recent["ci"]["index"]
        sep = r_lo > e_hi
        lines.append(f"- Era contrast of the index: 2015-2019 [{e_lo:.3f}, {e_hi:.3f}] vs "
                     f"2020-2024 [{r_lo:.3f}, {r_hi:.3f}] — "
                     + ("CIs separate (recent era materially larger)."
                        if sep else "CIs overlap; no resolved era difference."))
    if probe:
        excess = {}
        for era, er in probe["eras"].items():
            m, r = er["model"]["a_control"], er["reference"]["a_control"]
            if m is not None and r is not None:
                excess[era] = m - r
        if excess:
            parts = ", ".join(f"{era}: {v:+.3f}" for era, v in sorted(excess.items()))
            lines.append(f"- Fingerprint probe, model-minus-reference within-pair excess: "
                         f"{parts}.")
            rec = excess.get("2020-2024")
            if rec is not None and rec <= 0:
                lines.append("- The model shows NO within-pair discrimination edge over the "
                             "pure-code feature reference in 2020-2024 — the "
                             "fingerprint-recall signature is absent at pair level. "
                             "**Verdict: the era anomaly is not explained by memorization "
                             "detectable under protocol v1**; the era-asymmetric lift of the "
                             "flat baselines (random selection already lifts >1x in "
                             "2020-2024) points to era composition/base-rate structure "
                             "instead.")
            elif rec is not None:
                lines.append("- Positive recent-era excess over the feature reference — "
                             "**consistent with fingerprint recall**; escalate with a "
                             "jitter canary (protocol v2 candidate) before trusting "
                             "recent-era anonymized numbers.")
    return lines or ["- insufficient data for a verdict"]


def write_report(result: Mapping, probe: Mapping | None, spend: Mapping,
                 out_path: Path = RESULTS_MD) -> Path:
    g = result["gate"]
    lines = [
        "# Memorization canaries — canary protocol v1",
        "",
        f"Model: `{MODEL}` via Prime Intellect serverless, temp 0, naked (no harness), "
        f"over canary variants of the {len(result['seasons'])} ppr full_slate sets "
        f"(seasons {result['seasons'][0]}-{result['seasons'][-1]}). "
        "Protocol: same-position, ADP-adjacent, different-outcome pairs; the swapped arm "
        "exchanges ENTIRE feature packets between the pair (anon ids stable, swap recorded "
        "only in the sequestered canary KEY); the control arm is untouched. "
        "Ties score 0.5; 95% CIs bootstrap over seasons (10,000 resamples, seeded).",
        "",
        "## Canary sensitivity index = A_control − A_swap_pf",
        "",
        "A_control = pair accuracy on control pairs (higher p on the candidate that broke "
        "out). A_swap_pf = packet-following pair accuracy on swapped pairs (higher p on the "
        "question carrying the breakout packet, wherever the swap moved it). A packet-driven "
        "model scores index ≈ 0; a model betting on the hidden identity behind the anon id "
        "collapses on swapped pairs and scores index >> 0.",
        "",
        "| scope | A_control [95% CI] | n | A_swap_pf [95% CI] | n | index [95% CI] |",
        "|---|---|---|---|---|---|",
    ]

    def row(label: str, pooled: Mapping, ci: Mapping) -> str:
        return (f"| {label} | {_fmt(pooled['a_control'])} {_fmt_ci(ci.get('a_control'))} "
                f"| {pooled['n_control']} | {_fmt(pooled['a_swap_pf'])} "
                f"{_fmt_ci(ci.get('a_swap_pf'))} | {pooled['n_swap']} "
                f"| {_fmt(pooled['index'])} {_fmt_ci(ci.get('index'))} |")

    lines.append(row("pooled 2015-2024", result["pooled"], result["ci"]))
    for era, er in result["eras"].items():
        lines.append(row(f"era {era}", er["pooled"], er["ci"]))
    lines += [
        "",
        "## Canary gate (standard for every future checkpoint)",
        "",
        f"**Gate: FAIL iff the pooled index 95% CI lower bound > {GATE_INDEX_CI_LOW}.** "
        "A failing checkpoint's anonymized-track numbers are void (they measure id-recall, "
        "not reasoning) and cannot be claimed in any eval report. The gate runs with every "
        "BreakoutBench battery (`python evals/canaries.py run`) and its verdict line ships "
        "next to the score.",
        "",
        f"This run: **{'PASS' if g['pass'] else 'FAIL'}** — {g['reason']}.",
        "",
        "## Scope note + fingerprint probe (the era-anomaly question)",
        "",
        "A full-packet swap moves the profile fingerprint WITH the packet, so pretraining "
        "recall keyed on the packet content itself is swap-invariant: the swap index detects "
        "identity leakage through the anon-id channel (e.g. eval questions leaked into "
        "training data), not fingerprint recall. The fingerprint probe below addresses the "
        "era anomaly directly: within-pair accuracy on the UNSWAPPED originals (existing "
        "expanded-battery answers, no new inference) for the model vs the pure-code no-peek "
        "grounding reference. Pairs are ADP-adjacent so legitimate feature headroom is small "
        "and era-stable; model-over-reference excess concentrated in 2020-2024 is the "
        "fingerprint-recall signature.",
        "",
    ]
    if probe:
        lines += [
            "| era | Qwen naked pair acc [95% CI] | n | code reference pair acc [95% CI] | n |",
            "|---|---|---|---|---|",
        ]
        for era, er in probe["eras"].items():
            m, mci = er["model"], er["model_ci"]
            r, rci = er["reference"], er["reference_ci"]
            lines.append(f"| {era} | {_fmt(m['a_control'])} {_fmt_ci(mci.get('a_control'))} "
                         f"| {m['n_control']} | {_fmt(r['a_control'])} "
                         f"{_fmt_ci(rci.get('a_control'))} | {r['n_control']} |")
    else:
        lines.append("(original expanded-battery answers not on disk — probe skipped)")
    lines += ["", "## Verdict — era-anomaly hypothesis", ""]
    lines += verdict_lines(result, probe)
    lines += [
        "",
        "## Per-season detail",
        "",
        "| season | A_control | n | A_swap_pf | n |",
        "|---|---|---|---|---|",
    ]
    for s in result["seasons"]:
        ps = result["per_season"][s]
        lines.append(f"| {s} | {_fmt(ps['control'])} | {ps['n_control']} "
                     f"| {_fmt(ps['swap_pf'])} | {ps['n_swap']} |")
    lines += [
        "",
        f"Run cost: {spend.get('in', 0)}in/{spend.get('out', 0)}out tokens across "
        f"{spend.get('requests', 0)} requests ≈ ${spend.get('cost', 0.0):.3f} "
        f"(cap ${BUDGET_CAP:.2f}); failures: {len(spend.get('failures', []))}.",
        "",
    ]
    out_path.write_text("\n".join(lines))
    return out_path


# ---------------------------------------------------------------------------
# CLI


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Memorization canaries (protocol v1)")
    sub = ap.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="build canary question sets + keys")
    p_build.add_argument("--season", type=int, default=None)

    p_score = sub.add_parser("score", help="score canary answers files")
    p_score.add_argument("answers", nargs="+")
    p_score.add_argument("--reps", type=int, default=BOOTSTRAP_REPS)
    p_score.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)

    sub.add_parser("run", help="run naked Qwen over the canary sets, score, report")

    args = ap.parse_args(argv)
    if args.command == "build":
        if args.season == HOLDOUT_SEASON:
            ap.error(f"{HOLDOUT_SEASON} is the untouched eval holdout (GAMEPLAN §9)")
        seasons = [args.season] if args.season else list(SEASONS)
        build(seasons)
    elif args.command == "score":
        by_season = load_answers_files([Path(a) for a in args.answers])
        result = score_canaries(by_season, reps=args.reps, seed=args.seed)
        print(json.dumps(result, indent=1, default=str))
        print(f"\ngate: {'PASS' if result['gate']['pass'] else 'FAIL'} — "
              f"{result['gate']['reason']}")
    elif args.command == "run":
        from evals.run_expanded_qwen import append_ledger
        if not (QUESTIONS_DIR / f"canary_{SEASONS[0]}_{PRESET}.json").exists():
            build(SEASONS)
        spend = run_model()
        paths = sorted(ANSWERS_DIR.glob(f"qwen_canary_*_{PRESET}.json"))
        result = score_canaries(load_answers_files(paths))
        probe = fingerprint_probe()
        out = write_report(result, probe, spend)
        if spend["requests"]:
            append_ledger(spend, "Memorization canary run (protocol v1): naked Qwen over "
                                 f"{spend['requests']} canary ppr full_slate sets "
                                 "(evals/canaries.py)")
        print(out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
