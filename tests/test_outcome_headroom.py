from envs.draftgym import realistic_roster_points
from evals.draftbench import DraftPlayer
from evals.outcome_headroom import HindsightMarginalOracle, robust_metrics
from harness.league import LeagueConfig
from harness.scoring import PRESETS


def player(pid: str, pos: str, adp: float) -> DraftPlayer:
    return DraftPlayer(pid, pid, pos, "", adp, 1.0, 10, 0.0, None)


def test_robust_gate_requires_every_condition():
    passing = robust_metrics([1, 2, 3, -1, 1, 2, -1, 1, 2, -1])
    assert passing["passes_headroom_gate"] is True
    losing_more_often = robust_metrics([100, -1, -1, -1, 2])
    assert losing_more_often["mean_points_above_adp"] > 0
    assert losing_more_often["passes_headroom_gate"] is False


def test_hindsight_oracle_uses_realized_marginal_lineup_points():
    league = LeagueConfig(
        teams=2,
        roster={"QB": 1, "RB": 1, "WR": 1, "TE": 1, "FLEX": 0, "BENCH": 0},
        scoring=PRESETS["ppr"],
    )
    weak = player("weak", "RB", 1)
    strong = player("strong", "RB", 2)
    filler = [player("qb", "QB", 3), player("wr", "WR", 4), player("te", "TE", 5)]
    weekly = {
        "weak": {week: 1.0 for week in range(1, 18)},
        "strong": {week: 10.0 for week in range(1, 18)},
    }
    oracle = HindsightMarginalOracle(weekly, league)
    assert oracle.pick([weak, strong, *filler], (), league, 1) == "strong"
    assert realistic_roster_points((strong,), weekly, league) > realistic_roster_points(
        (weak,), weekly, league
    )
