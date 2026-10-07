"""
calibrate.py -- Checks how accurate the model's percentages really are,
then saves a correction that weekly_pipeline.py applies automatically.

How it works:
  1. Replays a past season (default 2025) week by week, weeks 4-18.
  2. For each week, builds projections using ONLY data available before
     that week's games (no peeking at the future).
  3. Checks who actually scored a touchdown that week.
  4. Groups players by predicted chance (0-10%, 10-20%, ...) and compares
     "what the model said" vs "what actually happened".
  5. Fits a simple correction and saves it to calibration.json.

Free data only -- uses zero Odds API requests. Takes a few minutes.

Usage:
    python calibrate.py
"""

from __future__ import annotations
import json
import numpy as np
import pandas as pd

from data_loader import compute_player_red_zone_usage, league_baseline_rates, team_red_zone_defense, PBP_COLUMNS
from td_model import build_weekly_projections
from share_model import build_share_projections

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
ROSTER_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.parquet"
PBP_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"


def load_pbp_with_scorers(season: int) -> pd.DataFrame:
    cols = list(dict.fromkeys(PBP_COLUMNS + ["season_type", "td_player_id"]))
    df = pd.read_parquet(PBP_URL_TMPL.format(season=season), columns=cols)
    return df[df["season_type"] == "REG"]


TD_GRID = {"decay": [1.0, 0.85, 0.7], "inj": [0.0, 0.25, 0.5, 0.75, 1.0], "opp": [0.0, 0.25, 0.5, 1.0]}


def collect_td(season: int = 2025, first_week: int = 4, last_week: int = 18) -> pd.DataFrame:
    """TD model ingredients for every player who played, weeks first..last, using only prior info."""
    from share_model import share_components
    from features import load_snaps, snap_shares, opponent_factors, out_players, INJURIES_URL, ROSTER_URL
    from yards_model import load_weekly_stats

    print(f"Loading {season}/{season - 1} play-by-play, stats, snaps, injuries, rosters...")
    pbp, prior_pbp = load_pbp_with_scorers(season), load_pbp_with_scorers(season - 1)
    stats = load_weekly_stats(season)
    stats["tds"] = stats["rushing_tds"].fillna(0) + stats["receiving_tds"].fillna(0)
    snaps = load_snaps(season)
    inj = pd.read_parquet(INJURIES_URL.format(season=season))
    rost = pd.read_parquet(ROSTER_URL.format(season=season))
    sched_all = pd.read_parquet(SCHEDULE_URL)

    rows = []
    for week in range(first_week, last_week + 1):
        r = rost[(rost["week"] == week - 1) & (rost["status"] == "ACT")]   # known before kickoff
        team_all = dict(zip(r["gsis_id"], r["team"]))
        pos_of = dict(zip(r["gsis_id"], r["position"]))
        out = out_players(season, week, inj)
        out_team = {p: team_all[p] for p in out if p in team_all}
        team_lookup = {p: t for p, t in team_all.items() if p not in out}
        sn = snap_shares(snaps, week)
        opp_td = opponent_factors(stats, week, "tds")

        wk_plays = pbp[pbp["week"] == week]
        scorers = set(wk_plays.loc[wk_plays["touchdown"] == 1, "td_player_id"].dropna())
        played = set(wk_plays["rusher_player_id"].dropna()) | set(wk_plays["receiver_player_id"].dropna())
        for decay in TD_GRID["decay"]:
            d = share_components(pbp, prior_pbp, sched_all, season, week, team_lookup, decay,
                                 out_team, sn, pos_of, opp_td)
            d = d[d["player_id"].isin(played)].copy()
            d["scored"], d["week"], d["decay"] = d["player_id"].isin(scorers).astype(int), week, decay
            rows.append(d)
        print(f"  week {week} done")
    return pd.concat(rows, ignore_index=True)


def td_prob(d: pd.DataFrame, p: dict) -> np.ndarray:
    from share_model import lambda_from_components
    return 1 - np.exp(-lambda_from_components(d, p))


def best_td_params(x: pd.DataFrame, fixed: dict | None = None):
    fixed = fixed or {}
    best = (None, 9.0)
    grid = {k: ([fixed[k]] if k in fixed else v) for k, v in TD_GRID.items()}
    for decay in grid["decay"]:
        xd = x[x["decay"] == decay]
        for inj in grid["inj"]:
            for opp in grid["opp"]:
                p = {"decay": decay, "inj": inj, "opp": opp}
                b = brier(td_prob(xd, p), xd["scored"])
                if b < best[1]:
                    best = (p, b)
    return best


def calibration_table(df: pd.DataFrame, prob_col: str = "model_prob") -> pd.DataFrame:
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%", "50-60%", "60-70%", "70%+"]
    d = df.copy()
    d["bucket"] = pd.cut(d[prob_col], bins=bins, labels=labels, include_lowest=True)
    return d.groupby("bucket", observed=True).agg(
        players=("scored", "size"),
        model_said=(prob_col, "mean"),
        actually_scored=("scored", "mean"),
    ).reset_index()


def fit_logistic_correction(p: np.ndarray, y: np.ndarray, iters: int = 50) -> tuple[float, float]:
    """
    Fits corrected_logit = a + b * logit(p) by Newton's method (Platt
    scaling). Plain numpy, no extra packages needed.
    """
    p = np.clip(p, 1e-4, 1 - 1e-4)
    x = np.log(p / (1 - p))
    X = np.column_stack([np.ones_like(x), x])
    w = np.array([0.0, 1.0])
    for _ in range(iters):
        z = X @ w
        mu = 1 / (1 + np.exp(-z))
        grad = X.T @ (y - mu)
        H = -(X.T * (mu * (1 - mu))) @ X
        step = np.linalg.solve(H, grad)
        w = w - step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(w[0]), float(w[1])


def apply_correction(p, a: float, b: float):
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    z = a + b * np.log(p / (1 - p))
    return 1 / (1 + np.exp(-z))


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


if __name__ == "__main__":
    data = collect_td(2025)
    train, test = data[data["week"] <= 11], data[data["week"] > 11]
    ref = test[test["decay"] == 1.0]
    base = brier(np.full(len(ref), train[train["decay"] == 1.0]["scored"].mean()), ref["scored"])

    def graded(fixed=None):
        p, _ = best_td_params(train, fixed)
        tr, te = train[train["decay"] == p["decay"]], test[test["decay"] == p["decay"]].copy()
        a, b = fit_logistic_correction(td_prob(tr, p), tr["scored"].values)
        return p, brier(apply_correction(td_prob(te, p), a, b), te["scored"])

    skill = lambda b: 100 * (1 - b / base)
    print("\n=== Anytime TD (graded on weeks 12-18) ===")
    _, b_old = graded({"decay": 1.0, "inj": 0.0, "opp": 0.0})
    p_all, b_all = graded()
    print(f"  current model:              {skill(b_old):5.1f}% better than guessing")
    print(f"  with new features:          {skill(b_all):5.1f}%   settings {p_all}")
    for name, fx in [("recent form", {"decay": 1.0}), ("injury boost", {"inj": 0.0}), ("opponent", {"opp": 0.0})]:
        _, b = graded(fx)
        print(f"    without {name:13s}      {skill(b):5.1f}%")

    # Injury boost graded only on teammates of injured players
    sub = test[(test["decay"] == 1.0) & (test["inj_add"] > 0.01)]
    if len(sub):
        r = "  ".join(f"{i}:{brier(td_prob(sub, {'decay': 1.0, 'inj': i, 'opp': 0.0}), sub['scored']):.4f}" for i in TD_GRID["inj"])
        print(f"  injured-teammate players ({len(sub)}): Brier by boost strength -> {r}")

    p_final, _ = best_td_params(data)
    xf = data[data["decay"] == p_final["decay"]]
    a_full, b_full = fit_logistic_correction(td_prob(xf, p_final), xf["scored"].values)
    with open("calibration.json", "w") as f:
        json.dump({"a": a_full, "b": b_full, "params": p_final, "model": "share",
                   "fit_on": "2025 weeks 4-18", "n_players": int(len(xf))}, f, indent=2)
    print(f"\nSaved calibration.json  (settings {p_final}, a={a_full:.3f}, b={b_full:.3f})")
