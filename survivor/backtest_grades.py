"""
backtest_grades.py -- Would your grading weights have helped in past seasons?

Uses the GRADE_WEIGHTS from survivor.py and replays 2016-2025 using only what
was known before each game (closing moneyline, that week's final injury
report, key players from earlier weeks, schedule/rest/travel).

Three checks:
  1. Beyond the line: when the grade says a team is better than its win
     chance (good injury/rest spot), do those teams actually win more than
     the line said?
  2. Weekly top pick: each week's #1 team by win chance vs. by grade.
  3. Survivor run: start week 1, pick the best unused team each week until
     the first loss. Line-only vs. your grades.

Run:  python backtest_grades.py      (takes a few minutes: downloads 10 seasons)
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ratings import load_schedule, moneyline_prob
from grading import SeasonData, injury_points, situation_points, DEFAULT_WEIGHTS
import survivor

SEASONS = range(2016, 2026)


def prev_away_table(sched):
    g = sched[["season", "week", "home_team", "away_team"]]
    long = pd.concat([g.rename(columns={"home_team": "team"}).assign(away=0)[["season", "week", "team", "away"]],
                      g.rename(columns={"away_team": "team"}).assign(away=1)[["season", "week", "team", "away"]]])
    long = long.sort_values(["team", "season", "week"])
    long["prev"] = long.groupby(["team", "season"])["away"].shift(1)
    return {(t, s, w): bool(p == 1) for t, s, w, p in zip(long.team, long.season, long.week, long.prev)}


def build_rows(W):
    sched = load_schedule()
    prev = prev_away_table(sched)
    rows = []
    for season in SEASONS:
        print(f"  {season}...", flush=True)
        sd = SeasonData(season)
        sg = sched[(sched.season == season) & sched.result.notna() & (sched.result != 0) &
                   sched.home_moneyline.notna() & sched.away_moneyline.notna()]
        for week, wk in sg.groupby("week"):
            hits = sd.injury_hits(week) if week > 1 else {}
            for r in wk.itertuples():
                ph = moneyline_prob(r.home_moneyline, r.away_moneyline)
                for team, opp, p, won in ((r.home_team, r.away_team, ph, r.result > 0),
                                          (r.away_team, r.home_team, 1 - ph, r.result < 0)):
                    inj = sum(x for _, x in injury_points(hits.get(team, []), W))
                    if W.get("mirror_opponent", True):
                        inj -= sum(x for _, x in injury_points(hits.get(opp, []), W))
                    sit = sum(x for _, x in situation_points(r, team, sched, W, prev_away=prev))
                    rows.append((season, week, team, opp, bool(r.div_game), p, inj, sit, int(won)))
    return pd.DataFrame(rows, columns=["season", "week", "team", "opp", "div", "p", "inj", "sit", "win"])


def logit_fit(X, y, offset):
    X = np.column_stack([np.ones(len(y)), X]); w = np.zeros(X.shape[1])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(offset + X @ w))); H = (X.T * (mu * (1 - mu))) @ X
        step = np.linalg.solve(H, X.T @ (y - mu)); w += step
        if np.abs(step).max() < 1e-9:
            break
    return w, np.sqrt(np.diag(np.linalg.inv(H)))


def main():
    W = {**DEFAULT_WEIGHTS, **survivor.GRADE_WEIGHTS}
    print("Replaying past seasons with your GRADE_WEIGHTS...")
    d = build_rows(W)
    d["grade"] = 100 * d.p + d.inj + d.sit
    print(f"\n{len(d) // 2} games, {min(SEASONS)}-{max(SEASONS)}.\n")

    # ---- 1. beyond the line (one row per game: home perspective is enough, but both sides are symmetric)
    off = np.log(d.p / (1 - d.p)).values
    print("1) DO THE EXTRA POINTS PREDICT WINS BEYOND THE BETTING LINE?")
    print("   Your grades treat 10 points like 10 percentage points of win chance.")
    print("   Here's what 10 points was actually worth in past games (for a 50/50 game):")
    for name, label in (("inj", "Injury points"), ("sit", "Rest/travel points")):
        x = d[name].values / 10.0
        if np.allclose(x, 0):
            print(f"   {label:20s} never triggered"); continue
        w, se = logit_fit(x[:, None], d.win.values, off)
        eff, lo, hi = 25 * w[1], 25 * (w[1] - 1.96 * se[1]), 25 * (w[1] + 1.96 * se[1])
        verdict = "REAL edge" if lo > 0 else ("works AGAINST you" if hi < 0 else "no clear effect (noise)")
        print(f"   {label:20s} {eff:+5.1f} pts of win%  (95% range {lo:+.1f} to {hi:+.1f})  -> {verdict}")
    print("   (Rows count each game twice, once per team, so the ranges are a bit too narrow.)")

    # ---- 2. weekly top pick
    print("\n2) EACH WEEK'S #1 TEAM (reuse allowed, division games skipped if AVOID_DIVISION):")
    pool = d[~d["div"]] if survivor.AVOID_DIVISION else d
    for key, label in (("p", "By win chance only"), ("grade", "By your grade")):
        top = pool.loc[pool.groupby(["season", "week"])[key].idxmax()]
        print(f"   {label:20s} won {top.win.mean():.1%} of {len(top)} weeks  (line expected {top.p.mean():.1%})")

    # ---- 3. survivor run
    print("\n3) SURVIVOR RUN: start week 1, best unused team each week, until the first loss")
    res = {}
    for key in ("p", "grade"):
        streaks, losses = [], []
        for season, s in pool.groupby("season"):
            used, alive, streak, lost = set(), True, 0, 0
            for week in sorted(s.week.unique()):
                c = s[(s.week == week) & ~s.team.isin(used)]
                if c.empty:
                    continue
                pick = c.loc[c[key].idxmax()]
                used.add(pick.team)
                if pick.win:
                    streak += alive
                else:
                    lost += 1; alive = False
            streaks.append(streak); losses.append(lost)
        res[key] = (np.mean(streaks), np.mean(losses), streaks)
    for key, label in (("p", "By win chance only"), ("grade", "By your grade")):
        m, l, st = res[key]
        print(f"   {label:20s} survived {m:4.1f} weeks on average  (by season: {st})"
              f"   | losses if you kept going: {l:.1f}/season")
    print("\nTen seasons is a small sample: a difference of a week or so can easily be luck.")
    print("Check (1) first -- it uses every game, so it's the most reliable test.")


if __name__ == "__main__":
    main()
