"""
yards_model.py -- Chance a player reaches a milestone (e.g. 100+ rush
yards, 6+ receptions). Built on OPPORTUNITIES, not past results:

  expected = opportunities per game  x  efficiency per opportunity
             (targets / carries /       (yards per target, catch rate...
              pass attempts)             pulled hard toward the position
                                         average, since a few big or bad
                                         games say little)

  Opportunities:  recent games count more (decay), blended with last
                  season (counts as PRIOR_WEIGHT_GAMES games), plus extra
                  opportunities handed over from injured teammates (inj).
  Adjustments:    team's expected points this week (beta) and how the
                  opponent does against his position (opp).
  Milestone odds: gamma distribution around the expected number (shape).

Settings are fit on 2025 by calibrate_yards.py -> calibration_yards.json.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import gamma

from features import recency_weights, opponent_factors, redistribute

STATS_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet"

# key -> (result column, opportunity column, min opps/game, efficiency shrink (in opps), label)
STATS = {
    "pass": ("passing_yards",   "attempts", 15.0, 150.0, "Pass Yds"),
    "rush": ("rushing_yards",   "carries",   3.0,  80.0, "Rush Yds"),
    "rec":  ("receiving_yards", "targets",   2.0,  50.0, "Rec Yds"),
    "recs": ("receptions",      "targets",   2.0,  40.0, "Receptions"),
}

PRIOR_WEIGHT_GAMES = 6.0
NO_PRIOR_WEIGHT_GAMES = 2.0
LEAGUE_AVG_POINTS = 22.0

DEFAULT_PARAMS = {s: {"shape": 2.5, "beta": 0.0, "decay": 0.85, "opp": 0.5, "inj": 0.6} for s in STATS}
DEFAULT_PARAMS["pass"]["shape"] = 8.0


def load_weekly_stats(season: int) -> pd.DataFrame:
    cols = ["player_id", "season", "week", "season_type", "team", "opponent_team", "position",
            "attempts", "passing_yards", "carries", "rushing_yards", "targets",
            "receiving_yards", "receptions", "rushing_tds", "receiving_tds"]
    d = pd.read_parquet(STATS_URL_TMPL.format(season=season), columns=cols)
    return d[d["season_type"] == "REG"].copy()


def opportunity_table(cur: pd.DataFrame, prior: pd.DataFrame, week: int, stat: str, decay: float) -> pd.DataFrame:
    """Per player: blended opportunities/game and shrunk efficiency (no adjustments yet)."""
    ycol, vcol, _, k_eff, _ = STATS[stat]
    c = cur[cur["week"] < week].copy()
    c["w"] = recency_weights(c["week"], week, decay)
    c["wvol"] = c["w"] * c[vcol]
    cg = c.groupby("player_id").agg(wvol=("wvol", "sum"), wgames=("w", "sum"), games=("week", "nunique"),
                                    vol=(vcol, "sum"), res=(ycol, "sum"), position=("position", "last"))
    pg = prior.groupby("player_id").agg(p_games=("week", "nunique"), p_vol=(vcol, "sum"),
                                        p_res=(ycol, "sum"), p_pos=("position", "last"))
    d = cg.join(pg, how="outer")
    for col in ["wvol", "wgames", "games", "vol", "res", "p_games", "p_vol", "p_res"]:
        d[col] = d[col].fillna(0.0)
    d["position"] = d["position"].fillna(d["p_pos"])
    has_prior = d["p_games"] > 0
    prior_pg = np.where(has_prior, d["p_vol"] / d["p_games"].clip(lower=1), 0.0)
    w_prior = np.where(has_prior, PRIOR_WEIGHT_GAMES, NO_PRIOR_WEIGHT_GAMES)
    d["vol_pg"] = (d["wvol"] + w_prior * prior_pg) / (d["wgames"] + w_prior)

    allp = pd.concat([c[["position", vcol, ycol]], prior[["position", vcol, ycol]]])
    sums = allp.groupby("position")[[ycol, vcol]].sum()
    pos_eff = sums[ycol] / sums[vcol].clip(lower=1)
    league_eff = allp[ycol].sum() / max(allp[vcol].sum(), 1)
    base_eff = d["position"].map(pos_eff).fillna(league_eff)
    d["eff"] = (d["res"] + d["p_res"] + k_eff * base_eff) / (d["vol"] + d["p_vol"] + k_eff)
    return d


def project_stat(table: pd.DataFrame, cur: pd.DataFrame, week: int, stat: str, params: dict,
                 team_of: dict, team_pts: dict, opponent_of: dict,
                 out: set | None = None, snaps: dict | None = None,
                 opp_cache: dict | None = None) -> pd.DataFrame:
    """Expected value of `stat` this week for every active, modelable player."""
    ycol, _, min_vol, _, _ = STATS[stat]
    out, snaps = out or set(), snaps or {}
    d = table
    pos_of = d["position"].to_dict()

    vols = {p: v for p, v in d["vol_pg"].items() if p in team_of}
    if out and params["inj"] > 0:
        # Out players keep their team mapping here so their volume can be handed over
        for p in out:
            if p in d.index and p in team_of:
                vols[p] = d.at[p, "vol_pg"]
        vols = redistribute(vols, team_of, pos_of, out, snaps, params["inj"])
    else:
        vols = {p: v for p, v in vols.items() if p not in out}

    if opp_cache is not None and ycol in opp_cache:
        opp_f = opp_cache[ycol]
    else:
        opp_f = opponent_factors(cur, week, ycol) if params["opp"] > 0 else {}

    rows = []
    for p, v in vols.items():
        team = team_of.get(p)
        if team is None or team not in team_pts:
            continue
        r = d.loc[p]
        if r["games"] + r["p_games"] < 2 or v < min_vol:
            continue
        mean = v * r["eff"]
        mean *= (team_pts[team] / LEAGUE_AVG_POINTS) ** params["beta"]
        mean *= opp_f.get((opponent_of.get(team), r["position"]), 1.0) ** params["opp"]
        rows.append((p, team, r["position"], v, r["eff"], max(mean, 0.1)))
    return pd.DataFrame(rows, columns=["player_id", "team", "position", "opps_pg", "eff", "mean"])


def prob_at_least(mean, threshold, shape: float):
    """P(result >= threshold); -0.5 because yards and catches are whole numbers."""
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-6)
    return gamma.sf(np.asarray(threshold, dtype=float) - 0.5, a=shape, scale=mean / shape)


def threshold_from_point(point: float) -> int:
    """'Over 99.5' and '100+' both mean 100 or more."""
    return int(math.ceil(point - 1e-9)) if point != int(point) else int(point)
