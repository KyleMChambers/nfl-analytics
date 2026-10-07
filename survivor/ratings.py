"""
ratings.py -- Win chances for any game, including weeks with no betting line yet.

  This week (and any week with a posted line): the moneyline, cut removed.
  Later weeks: "market power ratings". We work backwards from every
  point spread posted so far this season (recent ones count more) to
  get each team's strength in points, plus home-field advantage. Then
  any future game's spread = home rating - away rating + home field,
  turned into a win chance.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm

SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
SPREAD_SIGMA = 11.8     # points of randomness in an NFL game; fit to 2015-2025 moneylines
LINE_DECAY = 0.85       # each older week's lines count this much less
PRIOR_SEASON_WEIGHT = 0.25


INTL_WORDS = ("tottenham", "wembley", "stade de france", "bernabeu", "bayern", "allianz", "deutsche bank",
              "croke", "azteca", "estadio", "corinthians", "maracan", "melbourne", "mcg", "olympia", "london",
              "munich", "frankfurt", "berlin", "dublin", "madrid", "paris", "mexico", "sao paulo")


def load_schedule() -> pd.DataFrame:
    g = pd.read_parquet(SCHEDULE_URL)
    g = g[g["game_type"] == "REG"].copy()
    # International games are sometimes labeled as normal home games -- treat them all as neutral sites
    intl = g["stadium"].fillna("").str.lower().apply(lambda s: any(w in s for w in INTL_WORDS)) | \
           (g["gametime"].fillna("12:00") < "11:00")
    g.loc[intl, "location"] = "Neutral"
    return g


def moneyline_prob(home_ml, away_ml):
    """Home win chance from the two moneylines, with the book's cut removed."""
    def p(ml): return -ml / (-ml + 100) if ml < 0 else 100 / (ml + 100)
    ph, pa = p(home_ml), p(away_ml)
    return ph / (ph + pa)


def spread_to_prob(spread: float, sigma: float | None = None) -> float:
    """spread = points the home team is favored by."""
    return float(norm.cdf(spread / (sigma or SPREAD_SIGMA)))


def market_ratings(sched: pd.DataFrame, season: int, as_of_week: int, ridge: float = 2.0):
    """
    Team strength (points vs. an average team) from all spreads posted up to
    as_of_week (+1, since next week's lines are usually out), most recent counting most.
    Returns (ratings dict, home_field_points).
    """
    cur = sched[(sched["season"] == season) & (sched["week"] <= as_of_week + 1) & sched["spread_line"].notna()]
    prv = sched[(sched["season"] == season - 1) & (sched["week"] >= 13) & sched["spread_line"].notna()]
    rows = [(r, LINE_DECAY ** max(as_of_week + 1 - r.week, 0)) for r in cur.itertuples()]
    rows += [(r, PRIOR_SEASON_WEIGHT * LINE_DECAY ** (18 - r.week)) for r in prv.itertuples()]
    teams = sorted({t for r, _ in rows for t in (r.home_team, r.away_team)})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    X, y, w = [], [], []
    for r, wt in rows:
        x = np.zeros(n + 1)
        x[idx[r.home_team]], x[idx[r.away_team]] = 1, -1
        x[n] = 1.0 if r.location == "Home" else 0.0
        X.append(x); y.append(r.spread_line); w.append(wt)
    X, y, w = np.array(X), np.array(y), np.array(w)
    P = np.eye(n + 1) * ridge; P[n, n] = 0.0           # shrink ratings toward 0, not home field
    A = X.T @ (X * w[:, None]) + P
    beta = np.linalg.solve(A, X.T @ (w * y))
    ratings = {t: beta[i] - beta[:n].mean() for t, i in idx.items()}
    return ratings, float(beta[n])


def game_prob(row, ratings: dict, hfa: float, use_market: bool = True):
    """(home win chance, source, home-favored-by points) for one schedule row."""
    if use_market and pd.notna(row.home_moneyline) and pd.notna(row.away_moneyline):
        p = moneyline_prob(row.home_moneyline, row.away_moneyline)
        return p, "moneyline", (row.spread_line if pd.notna(row.spread_line) else np.nan)
    if use_market and pd.notna(row.spread_line):
        return spread_to_prob(row.spread_line), "spread", row.spread_line
    spread = ratings.get(row.home_team, 0.0) - ratings.get(row.away_team, 0.0) + (hfa if row.location == "Home" else 0.0)
    return spread_to_prob(spread), "projected", spread
