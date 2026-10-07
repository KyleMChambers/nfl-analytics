"""Weekly driver.

    python -m nflelo ratings                  # current power ratings + unit splits
    python -m nflelo spreads                  # spreads for the next unplayed week
    python -m nflelo spreads --week 5         # a specific week
    python -m nflelo backtest --from 2018     # how the model has held up
    python -m nflelo fit                      # re-tune parameters (slow, overfits easily)
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import pandas as pd

from . import config as C
from . import data, ratings, backtest, pfr

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def build(seasons, through=None, use_pfr=False, use_ftn=True, cfg=C):
    pbp = data.load_pbp(seasons)
    units = data.team_game_units(pbp)
    if use_ftn:
        ftn = data.load_ftn(seasons)
        units = data.attach_ftn(units, data.ftn_team_game(ftn, pbp) if ftn is not None else None)
    games = data.load_games()
    games = games[games.season >= min(seasons)]
    model, hist = ratings.run(units, games, through=through, cfg=cfg)
    if use_pfr and through:
        if not pfr.blend_into(model, through[0]):
            print("[warn] PFR pressure fetch failed; using play-by-play proxy",
                  file=sys.stderr)
    return model, hist, games, units


def current_injuries(season, week, min_snap_share=0.15):
    """{team: [(position, status, snap_share), ...]} for this week's report,
    using each player's average snap share over the last few weeks as the
    'how much do they matter' proxy. Returns {} on any fetch failure --
    injuries are an optional refinement, never load-bearing."""
    inj = data.load_injuries([season])
    snaps = data.load_snap_counts([season])
    if inj is None or snaps is None:
        return {}
    wk = inj[(inj.season == season) & (inj.week == week)
             & inj.report_status.isin(["Out", "Doubtful", "Questionable"])]
    if wk.empty:
        return {}
    lb = max(week - C.INJURY_SNAP_LOOKBACK_WEEKS, 1)
    recent = snaps[(snaps.season == season) & (snaps.week >= lb) & (snaps.week < week)].copy()
    recent["team"] = data.standardize(recent["team"])
    # No shared player id between the injuries and snap-counts tables; join
    # on (team, full name), which is imperfect on suffixes/nicknames but
    # good enough for a "how much does this player matter" weight.
    recent["_key"] = recent["team"] + "|" + recent["player"].str.lower().str.strip()
    snap_lookup = recent.groupby("_key")[["offense_pct", "defense_pct"]].mean()

    out = {}
    for row in wk.itertuples(index=False):
        team = data.standardize(pd.Series([row.team])).iloc[0]
        key = f"{team}|{str(row.full_name).lower().strip()}"
        share = np.nan
        if key in snap_lookup.index:
            r = snap_lookup.loc[key]
            share = max(r.get("offense_pct", 0) or 0, r.get("defense_pct", 0) or 0)
        if pd.isna(share) or share == 0:
            share = 0.6  # no snap history found for a reported player: assume meaningful
        out.setdefault(team, []).append((row.position, row.report_status, share))
    return out


def next_week(games):
    """First week that still has unplayed games -- not max(played)+1, which
    skips the rest of the current week once the Thursday game is final."""
    season = int(games[games.result.notna()].season.max())
    cur = games[games.season == season]
    unplayed = cur[cur.result.isna()]
    if unplayed.empty:
        return season, int(cur.week.max())
    return season, int(unplayed.week.min())


def cmd_ratings(args):
    games_all = data.load_games()
    season, week = (args.season, args.week) if args.week else next_week(games_all)
    seasons = range(season - args.history, season + 1)
    model, _, _, _ = build(seasons, through=(season, week), use_pfr=args.pfr, use_ftn=not args.no_ftn)
    t = model.rating_table()
    t["elo"] = t["elo"].round(1)
    for c in ("power_rating", "pass_off_pts", "rush_off_pts", "pass_def_pts", "rush_def_pts"):
        t[c] = t[c].round(2)
    for c in ("pass_off_epa", "rush_off_epa", "pass_def_epa", "rush_def_epa",
              "pressure_gen", "pressure_allowed", "qb_rating",
              "qb_fault_sack_rate", "blitz_faced", "blitz_generated"):
        t[c] = t[c].astype(float).round(4)
    out = args.out or f"ratings_{season}_wk{week}.csv"
    t.to_csv(out, index=False)
    cols = ["rank", "team", "elo", "power_rating", "pass_off_pts", "rush_off_pts",
            "pass_def_pts", "rush_def_pts", "pressure_gen", "games"]
    print(f"\nPower ratings entering {season} Week {week}")
    print("(points = expected margin vs a league-average team, neutral field)\n")
    print(t[cols].to_string(index=False))
    print(f"\nwrote {out}")


def cmd_spreads(args):
    games_all = data.load_games()
    season, week = (args.season, args.week) if args.week else next_week(games_all)
    seasons = range(season - args.history, season + 1)
    model, _, games, _ = build(seasons, through=(season, week), use_pfr=args.pfr, use_ftn=not args.no_ftn)

    slate = games_all[(games_all.season == season) & (games_all.week == week)]
    if not args.include_played:
        slate = slate[slate.result.isna()]
    if slate.empty:
        sys.exit(f"no games found for {season} week {week}")

    injuries_by_team = {}
    if not args.no_injuries:
        try:
            injuries_by_team = current_injuries(season, week)
        except Exception as e:
            print(f"[warn] injury lookup failed, skipping: {e}", file=sys.stderr)

    rows = []
    for gm in slate.itertuples(index=False):
        margin, parts = model.predict(
            gm.home_team, gm.away_team,
            neutral=(str(gm.location).lower() == "neutral"),
            home_rest=gm.home_rest, away_rest=gm.away_rest,
            div_game=bool(gm.div_game),
            home_qb=gm.home_qb_id, away_qb=gm.away_qb_id,
            roof=getattr(gm, "roof", None),
            home_injuries=injuries_by_team.get(gm.home_team, ()),
            away_injuries=injuries_by_team.get(gm.away_team, ()))
        fav, dog = (gm.home_team, gm.away_team) if margin >= 0 else (gm.away_team, gm.home_team)
        rows.append({
            "game_id": gm.game_id, "gameday": gm.gameday,
            "away": gm.away_team, "home": gm.home_team,
            "proj_home_margin": round(margin, 2),
            "my_line": f"{fav} -{abs(margin):.1f}",
            "key_number": _key_note(abs(margin)),
            "home_qb": gm.home_qb_name, "away_qb": gm.away_qb_name,
            **{k: round(v, 2) for k, v in parts.items()},
        })
    df = pd.DataFrame(rows)
    out = args.out or f"spreads_{season}_wk{week}.csv"
    df.to_csv(out, index=False)
    show = ["gameday", "away", "home", "my_line", "proj_home_margin",
            "base", "hfa", "qb", "pressure", "qb_fault_sack", "blitz",
            "injuries", "rest", "key_number"]
    print(f"\nProjected spreads — {season} Week {week}\n")
    print(df[show].to_string(index=False))
    print(f"\nwrote {out}")


def _key_note(m):
    """Flag lines sitting on or near the numbers that decide games."""
    for k, label in ((3, "ON 3"), (7, "ON 7"), (10, "ON 10"), (6, "ON 6"), (4, "ON 4")):
        if abs(m - k) < 0.25:
            return label
        if abs(m - k) < 0.75:
            return f"near {k}"
    return ""


def cmd_backtest(args):
    seasons = range(args.start, args.end + 1)
    pbp = data.load_pbp(seasons)
    units = data.team_game_units(pbp)
    games = data.load_games()
    games = games[games.season >= args.start]
    s, hist = backtest.score(units, games, backtest.Cfg(), eval_from=getattr(args, "from"))
    print("\nBacktest", f"{getattr(args,'from')}-{args.end}")
    for k, v in s.items():
        print(f"  {k:20s} {v:.4f}" if isinstance(v, float) else f"  {k:20s} {v}")
    if args.out:
        hist.to_csv(args.out, index=False)
        print(f"wrote {args.out}")


def cmd_fit(args):
    seasons = range(args.start, args.end + 1)
    pbp = data.load_pbp(seasons)
    units = data.team_game_units(pbp)
    games = data.load_games()
    games = games[games.season >= args.start]
    grid = {
        "W_PRESSURE": [0.0, 0.05, 0.1, 0.2, 0.35],
        "W_THIRD_DOWN": [0.0, 0.2, 0.5, 1.0, 2.0],
        "W_TURNOVER_LUCK": [0.0, 0.5, 1.0],
        "W_QB": [0.0, 0.25, 0.5, 0.75, 1.0],
        "W_COACH": [0.0, 0.1, 0.3],
        "K_PASS_OFF": [0.02, 0.035, 0.05, 0.065],
        "K_PASS_DEF": [0.01, 0.015, 0.03, 0.045],
        "K_RUSH_OFF": [0.03, 0.045, 0.06, 0.075],
        "K_RUSH_DEF": [0.02, 0.035, 0.05, 0.065],
        "CARRYOVER_PASS": [0.5, 0.62, 0.7, 0.8],
        "CARRYOVER_RUSH": [0.35, 0.5, 0.6],
        "HOME_FIELD_POINTS": [1.2, 1.5, 1.65, 1.9],
        "EARLY_K_MULT": [1.0, 1.5, 2.0],
    }
    cfg, best = backtest.fit(units, games, grid, passes=3,
                             eval_from=getattr(args, "from"), eval_to=args.holdout - 1)
    print("\nfitted on", getattr(args, "from"), "-", args.holdout - 1)
    for k in grid:
        print(f"  {k} = {getattr(cfg, k)}")
    te, _ = backtest.score(units, games, cfg, args.holdout, 9999)
    print("\nHOLDOUT", args.holdout, "+:",
          {k: round(v, 4) if isinstance(v, float) else v for k, v in te.items()})
    print("\nCopy the values above into config.py only if the holdout numbers "
          "hold up. A fit that improves training RMSE and worsens holdout RMSE "
          "is the model memorizing, not learning.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="nflelo")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, fn in (("ratings", cmd_ratings), ("spreads", cmd_spreads)):
        s = sub.add_parser(name)
        s.add_argument("--season", type=int, default=None)
        s.add_argument("--week", type=int, default=None)
        s.add_argument("--history", type=int, default=5,
                       help="prior seasons of data to warm the ratings on")
        s.add_argument("--pfr", action="store_true",
                       help="blend in charted pressure rates from PFR")
        s.add_argument("--no-ftn", action="store_true",
                       help="skip FTN charting (qb-fault sacks, blitz rate)")
        s.add_argument("--out", default=None)
        s.add_argument("--include-played", action="store_true",
                       help="also project games already final this week")
        s.add_argument("--no-injuries", action="store_true",
                       help="(spreads only) skip the injury-report adjustment")
        s.set_defaults(func=fn)

    b = sub.add_parser("backtest")
    b.add_argument("--start", type=int, default=2015)
    b.add_argument("--end", type=int, default=2026)
    b.add_argument("--from", type=int, default=2018, dest="from")
    b.add_argument("--out", default=None)
    b.set_defaults(func=cmd_backtest)

    f = sub.add_parser("fit")
    f.add_argument("--start", type=int, default=2015)
    f.add_argument("--end", type=int, default=2026)
    f.add_argument("--from", type=int, default=2018, dest="from")
    f.add_argument("--holdout", type=int, default=2024)
    f.set_defaults(func=cmd_fit)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
