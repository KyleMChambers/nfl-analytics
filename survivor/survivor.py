"""
survivor.py -- Weekly NFL survivor pick helper.

How a pick is graded (0-100, A = 90+):
  1. WIN CHANCE: from the moneyline (or spread), sportsbook's cut removed.
  2. FUTURE COST: using a team now that you'll badly need later costs points.
  3. INJURIES: your key players out (QB, starting OL, top WR/RB, top pass
     rushers, starting CBs) cost points; the opponent's give you points.
  4. REST / TRAVEL: rest edge, opponent on back-to-back road games, long
     time-zone trips, West Coast teams in 1 p.m. ET games.
  Every point value is in GRADE_WEIGHTS below. See grading.py.
  Check how your weights would have done historically: python backtest_grades.py

Run:  python survivor.py     (no API key needed -- all free data)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# =====================================================================
#  YOUR SETTINGS
# =====================================================================
SEASON = 2026
WEEK = 5                     # the week you're picking for
USED_TEAMS = []              # teams you've already picked, e.g. ["BUF", "PHI", "DET", "KC"]
AVOID_DIVISION = True        # skip division games (data says big favorites are just as safe in them)
PLAN_WEEKS = 8               # how many weeks ahead to plan
SHOW = 6                     # how many candidates to show in detail

# Grading points (positive = good for the team you're picking). Defaults live in grading.py.
GRADE_WEIGHTS = {
    "qb": 15,              # starting QB out
    "ol": 6,               # each starting offensive lineman out
    "ol_two_plus": 6,      # extra when 2+ starting linemen are out
    "skill": 5,            # each top-2 WR / top RB out
    "pass_rusher": 4,      # each top-2 pass rusher out
    "cb": 4,               # each starting cornerback out
    "questionable": 0.5,   # Questionable = half penalty (Out/Doubtful = full)
    "mirror_opponent": True,  # opponent's injured key players ADD the same points to you
    "rest_per_day": 3,     # per extra day of rest vs. the opponent ...
    "rest_cap": 12,        # ... up to this many points
    "b2b_road": 8,         # opponent on back-to-back road games (you on one = minus)
    "time_zones": 5,       # road team crossing 2+ time zones
    "early_west": 5,       # West/Mountain road team in a 1 p.m. ET game
    "base_offset": 10,     # starting score = win% + 10  (80% favorite starts at 90)
    "future_per_pct": 0.5, # points off per 1% of later survival odds you give up
}
# =====================================================================

from ratings import load_schedule, market_ratings, game_prob
from context import Context
from grading import SeasonData, injury_points, situation_points, grade_team, DEFAULT_WEIGHTS

TINY = 1e-6  # stand-in win chance for "can't use" (bye week, division game you're avoiding)


def team_week_probs(sched, season, week_from, week_to, ratings, hfa):
    """{(team, week): (win chance, source, spread for team, opponent, home?, division?, row)}"""
    out = {}
    g = sched[(sched.season == season) & sched.week.between(week_from, week_to)]
    for r in g.itertuples():
        p_home, src, spread = game_prob(r, ratings, hfa)
        sp = spread if pd.notna(spread) else np.nan
        out[(r.home_team, r.week)] = (p_home, src, sp, r.away_team, True, bool(r.div_game), r)
        out[(r.away_team, r.week)] = (1 - p_home, src, -sp, r.home_team, False, bool(r.div_game), r)
    return out


def best_plan(teams, weeks, probs, exclude=()):
    """Best one-team-per-week path (max chance of surviving every week). Returns (log survival, {week: team})."""
    pool = [t for t in teams if t not in exclude]
    if not weeks or not pool:
        return 0.0, {}
    cost = np.zeros((len(pool), len(weeks)))
    for i, t in enumerate(pool):
        for j, w in enumerate(weeks):
            v = probs.get((t, w))
            p = TINY if v is None or (AVOID_DIVISION and v[5]) else v[0]
            cost[i, j] = -np.log(max(p, TINY))
    rows, cols = linear_sum_assignment(cost)
    return -cost[rows, cols].sum(), {weeks[c]: pool[r] for r, c in zip(rows, cols)}


def run():
    sched = load_schedule()
    used = {t.upper() for t in USED_TEAMS}
    print(f"Survivor helper -- {SEASON} week {WEEK}   (used so far: {', '.join(sorted(used)) or 'none'})\n")

    ratings, hfa = market_ratings(sched, SEASON, WEEK - 1)
    last = min(18, WEEK + PLAN_WEEKS)
    probs = team_week_probs(sched, SEASON, WEEK, last, ratings, hfa)
    all_teams = sorted({t for (t, _w) in probs})
    avail = [t for t in all_teams if t not in used]
    future_weeks = list(range(WEEK + 1, last + 1))

    W = {**DEFAULT_WEIGHTS, **GRADE_WEIGHTS}
    print("Loading key players and injury report...")
    sd = SeasonData(SEASON)
    inj_hits = sd.injury_hits(WEEK)
    if not sd.has_report(WEEK):
        print(f"  [note] week {WEEK} injury report isn't posted yet, so injuries aren't in the grades.\n"
              f"         Re-run Thursday-Saturday (final statuses usually come out Friday).")
    print()

    base_future, _ = best_plan(avail, future_weeks, probs)
    cands = []
    for t in avail:
        v = probs.get((t, WEEK))
        if v is None:
            continue  # bye
        p, src, spread, opp, home, div, row = v
        if AVOID_DIVISION and div:
            continue
        fut, plan = best_plan(avail, future_weeks, probs, exclude={t})
        cost = base_future - fut                       # lost future log-survival from using t now
        future_pct = 100 * (1 - np.exp(-cost)) if cost > 0 else 0.0
        g = grade_team(p, future_pct, inj_hits.get(t, []), inj_hits.get(opp, []),
                       situation_points(row, t, sched, W), W)
        cands.append(dict(team=t, opp=opp, home=home, p=p, src=src, spread=spread, cost=cost,
                          score=g["score"], g=g, plan=plan, row=row, fut=fut))
    if not cands:
        print("No available teams this week with your settings. Try AVOID_DIVISION = False.")
        return
    cands.sort(key=lambda c: c["score"], reverse=True)

    print("RANKED PICKS  (Inj = injury points, Rest/Trv = rest & travel points,")
    print("              Future = how much using them now lowers your odds of surviving later weeks)\n")
    print(f"  {'#':>2} {'Grade':5} {'Score':>5}  {'Team':5} {'vs':>3} {'Opp':5} {'Win%':>6} {'Spread':>7} "
          f"{'Inj':>5} {'Rest/Trv':>8} {'Future':>7}")
    for i, c in enumerate(cands[:15], 1):
        loc = "vs" if c["home"] else "@"
        sp = f"{c['spread']:+.1f}" if pd.notna(c["spread"]) else "  --"
        fc = f"-{(1 - np.exp(-c['cost'])):.0%}" if c["cost"] > 0.005 else "none"
        print(f"  {i:>2} {c['g']['grade']:5} {c['score']:5.1f}  {c['team']:5} {loc:>3} {c['opp']:5} {c['p']:6.1%} {sp:>7} "
              f"{c['g']['inj']:+5.1f} {c['g']['sit']:+8.1f} {fc:>7}")

    ctx = Context(sched, SEASON, WEEK)
    print("\n" + "=" * 78)
    print("DETAIL ON THE TOP PICKS")
    print("=" * 78)
    for i, c in enumerate(cands[:SHOW], 1):
        t, o, r = c["team"], c["opp"], c["row"]
        home_t, away_t = r.home_team, r.away_team
        if r.location == "Home":
            lo, wx = ctx.weather(home_t, str(r.gameday))
        else:
            lo, wx = None, f"neutral site ({r.stadium})"
        tr = ctx.travel(away_t, home_t) if r.location == "Home" else None
        qb_t, qb_o = ctx.qb.get(t), ctx.qb.get(o)
        coach_t = r.home_coach if c["home"] else r.away_coach
        coach_o = r.away_coach if c["home"] else r.home_coach
        print(f"\n#{i}  {t} {'vs' if c['home'] else '@'} {o}  --  GRADE {c['g']['grade']} ({c['score']:.1f})"
              f"  |  {r.weekday} {r.gameday} {r.gametime}")
        print("    How the grade adds up:")
        for txt, pts in c["g"]["parts"]:
            print(f"      {pts:+6.1f}  {txt}")
        print(f"    Records:        {t} {ctx.record(t):7} | {o} {ctx.record(o)}")
        print(f"    Points/game:    {t} scores {ctx.ppg(t):4.1f}, allows {ctx.papg(t):4.1f} | "
              f"{o} scores {ctx.ppg(o):4.1f}, allows {ctx.papg(o):4.1f}")
        print(f"    Yards allowed:  {t} D {ctx.def_ypg.get(t, float('nan')):5.0f}/game | {o} D {ctx.def_ypg.get(o, float('nan')):5.0f}/game")
        print(f"    QB rating:      {t}: {qb_t[0]} {qb_t[1]:.1f}" if qb_t else f"    QB rating:      {t}: --", end="")
        print(f" | {o}: {qb_o[0]} {qb_o[1]:.1f}" if qb_o else f" | {o}: --")
        print(f"    Head coaches:   {ctx.coach_record(coach_t)} | {ctx.coach_record(coach_o)}   (since 1999)")
        print(f"    Weather:        {wx}")
        rest_t = r.home_rest if c["home"] else r.away_rest
        rest_o = r.away_rest if c["home"] else r.home_rest
        print(f"    Rest days:      {t} {rest_t} | {o} {rest_o}")

        flags = []
        if c["home"] and ctx.prev_was_road(o):
            flags.append(f"{o} on back-to-back road games")
        if tr:
            miles, tz = tr
            if miles > 1500 or tz >= 2:
                flags.append(f"{away_t} traveling {miles:,.0f} miles / {tz} time zone(s)")
        cf = ctx.climate_flag(away_t, home_t, r.gameday, lo) if r.location == "Home" else None
        if cf:
            flags.append(cf)
        if r.div_game:
            flags.append("DIVISION GAME")
        if r.location != "Home":
            flags.append(f"neutral site: {r.stadium}")
        print(f"    Spot / travel:  {'; '.join(flags) if flags else 'nothing notable'}")
        for team in (t, o):
            if ctx.inj.get(team):
                print(f"    Injuries {team:4}:  " + "; ".join(ctx.inj[team][:5]))
        if c["cost"] > 0.005:
            print(f"    Saving them:    using {t} now lowers your odds of surviving weeks {WEEK+1}-{last} "
                  f"by {(1 - np.exp(-c['cost'])):.0%} (they're valuable later)")

    top = cands[0]
    print("\n" + "=" * 78)
    print(f"SUGGESTED PICK: {top['team']}  grade {top['g']['grade']} ({top['score']:.1f}), {top['p']:.1%} to win")
    plan = top["plan"]
    if plan:
        path = ", ".join(f"wk{w} {plan[w]}" for w in sorted(plan))
        surv = np.exp(top["fut"])
        print(f"Rough plan after that (projected, will change as lines come out): {path}")
        print(f"Chance of surviving this week AND every week through {last} on that plan: {top['p'] * surv:.1%}")
    print("Before locking it in: check the latest injury news for both teams.")


if __name__ == "__main__":
    run()
