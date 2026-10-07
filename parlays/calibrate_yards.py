"""
calibrate_yards.py -- Grades the yards/receptions model on 2025 and saves
its settings to calibration_yards.json.

  - Replays 2025 weeks 4-18 using only info available before each week
    (stats, snap counts, injury report, betting lines).
  - Fits each feature's strength on weeks 4-11, grades on weeks 12-18.
  - Checks each feature by switching it off, and compares to the old
    yards-per-game model on the exact same players.
  - Refits on all weeks and saves the settings for live use.

Free data only, zero Odds API requests. Usage: python calibrate_yards.py
"""

from __future__ import annotations
import itertools
import json
import numpy as np
import pandas as pd

from yards_model import STATS, load_weekly_stats, opportunity_table, prob_at_least, LEAGUE_AVG_POINTS
from features import load_snaps, snap_shares, opponent_factors, out_players, redistribute, INJURIES_URL, ROSTER_URL
from share_model import implied_team_points
from calibrate import brier

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
THRESHOLDS = {
    "pass": [200, 225, 250, 275, 300, 325, 350],
    "rush": [40, 50, 60, 70, 80, 90, 100, 125, 150],
    "rec":  [40, 50, 60, 70, 80, 90, 100, 125, 150],
    "recs": [2, 3, 4, 5, 6, 7, 8, 10],
}
GRID = {
    "decay": [1.0, 0.85, 0.7],
    "inj":   [0.0, 0.5, 1.0],
    "opp":   [0.0, 0.5, 1.0],
    "beta":  [0.0, 0.5, 1.0],
    "shape": [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0],
}


def collect(season: int = 2025, first_week: int = 4, last_week: int = 18) -> pd.DataFrame:
    """One row per (player who played, week, stat, decay) with every ingredient stored separately."""
    print(f"Loading {season}/{season - 1} stats, snaps, injuries, rosters, schedule...")
    cur, prior = load_weekly_stats(season), load_weekly_stats(season - 1)
    snaps = load_snaps(season)
    inj = pd.read_parquet(INJURIES_URL.format(season=season))
    rost = pd.read_parquet(ROSTER_URL.format(season=season))
    sched = pd.read_parquet(SCHEDULE_URL)
    rows = []
    for week in range(first_week, last_week + 1):
        pts = implied_team_points(sched, season, week)
        wk = sched[(sched["season"] == season) & (sched["week"] == week)]
        opp_of = {**dict(zip(wk["home_team"], wk["away_team"])), **dict(zip(wk["away_team"], wk["home_team"]))}
        # previous week's roster = what you'd know before kickoff (game-day file already marks injured as inactive)
        r = rost[(rost["week"] == week - 1) & (rost["status"] == "ACT")]
        team_of = dict(zip(r["gsis_id"], r["team"]))
        out = out_players(season, week, inj)
        sn = snap_shares(snaps, week)
        played = cur[cur["week"] == week].set_index("player_id")
        for stat, (ycol, _, min_vol, _, _) in STATS.items():
            oppf = opponent_factors(cur, week, ycol)
            for decay in GRID["decay"]:
                t = opportunity_table(cur, prior, week, stat, decay)
                pos_of = t["position"].to_dict()
                base = {p: v for p, v in t["vol_pg"].items() if p in team_of}
                full = redistribute(base, team_of, pos_of, out, sn, 1.0)  # fraction=1; scaled later
                for p, v_full in full.items():
                    if p not in played.index:
                        continue
                    tr = t.loc[p]
                    if tr["games"] + tr["p_games"] < 2 or v_full < min_vol:
                        continue
                    team = team_of[p]
                    if team not in pts:
                        continue
                    old_mean = (tr["res"] + 6.0 * (tr["p_res"] / tr["p_games"] if tr["p_games"] else 0)) / \
                               (tr["games"] + (6.0 if tr["p_games"] else 2.0))
                    rows.append((stat, week, decay, p, base[p], v_full - base[p], tr["eff"], pts[team],
                                 oppf.get((opp_of.get(team), tr["position"]), 1.0), old_mean,
                                 played.at[p, ycol] if np.isscalar(played.at[p, ycol]) else played.loc[p, ycol].iloc[0]))
        print(f"  week {week} done")
    return pd.DataFrame(rows, columns=["stat", "week", "decay", "player_id", "base_vol", "inj_add", "eff",
                                       "team_pts", "opp_raw", "old_mean", "actual"])


def expand(df: pd.DataFrame, stat: str) -> pd.DataFrame:
    d = df[df["stat"] == stat]
    return pd.concat([d.assign(threshold=t, hit=(d["actual"] >= t).astype(int)) for t in THRESHOLDS[stat]],
                     ignore_index=True)


def predict(d: pd.DataFrame, p: dict) -> np.ndarray:
    mean = (d["base_vol"].values + p["inj"] * d["inj_add"].values) * d["eff"].values
    mean = mean * (d["team_pts"].values / LEAGUE_AVG_POINTS) ** p["beta"] * d["opp_raw"].values ** p["opp"]
    return prob_at_least(mean, d["threshold"].values, p["shape"])


def best_params(x: pd.DataFrame, fixed: dict | None = None) -> tuple[dict, float]:
    fixed = fixed or {}
    best = (None, 9.0)
    for decay in ([fixed["decay"]] if "decay" in fixed else GRID["decay"]):
        xd = x[x["decay"] == decay]
        for inj, opp, beta in itertools.product(*[[fixed[k]] if k in fixed else GRID[k] for k in ("inj", "opp", "beta")]):
            for shape in GRID["shape"]:
                p = {"decay": decay, "inj": inj, "opp": opp, "beta": beta, "shape": shape}
                b = brier(predict(xd, p), xd["hit"])
                if b < best[1]:
                    best = (p, b)
    return best


def old_model_brier(train, test):
    """Old model (yards per game, no opportunities/recency/opponent/injuries), same players."""
    tr, te = train[train["decay"] == 1.0], test[test["decay"] == 1.0]
    best = (None, 9.0)
    for beta in GRID["beta"]:
        for shape in GRID["shape"]:
            b = brier(prob_at_least(tr["old_mean"] * (tr["team_pts"] / LEAGUE_AVG_POINTS) ** beta, tr["threshold"], shape), tr["hit"])
            if b < best[1]:
                best = ((beta, shape), b)
    beta, shape = best[0]
    return brier(prob_at_least(te["old_mean"] * (te["team_pts"] / LEAGUE_AVG_POINTS) ** beta, te["threshold"], shape), te["hit"])


if __name__ == "__main__":
    data = collect(2025)
    saved = {}
    for stat in STATS:
        x = expand(data, stat)
        train, test = x[x["week"] <= 11], x[x["week"] > 11]
        ref = test[test["decay"] == 1.0]
        base = brier(ref["threshold"].map(train[train["decay"] == 1.0].groupby("threshold")["hit"].mean()).values, ref["hit"])
        skill = lambda b: 100 * (1 - b / base)

        def graded(fixed=None):
            p, _ = best_params(train, fixed)
            te = test[test["decay"] == p["decay"]]
            return p, brier(predict(te, p), te["hit"])

        p_all, b_all = graded()
        print(f"\n=== {STATS[stat][4]} (graded on weeks 12-18) ===")
        print(f"  old model:                  {skill(old_model_brier(train, test)):5.1f}% better than guessing")
        print(f"  new model (all features):   {skill(b_all):5.1f}%   settings {p_all}")
        for name, fx in [("recent form", {"decay": 1.0}), ("injury boost", {"inj": 0.0}), ("opponent", {"opp": 0.0})]:
            _, b = graded(fx)
            print(f"    without {name:13s}      {skill(b):5.1f}%")

        p_final, _ = best_params(x)  # refit on all weeks for live use
        saved[stat] = p_final
    saved["fit_on"] = "2025 weeks 4-18"
    with open("calibration_yards.json", "w") as f:
        json.dump(saved, f, indent=2)
    print("\nSaved calibration_yards.json")
