"""
props_pipeline.py -- Weekly value finder for BetMGM player props:
Anytime TD + milestones (pass/rush/rec yards, receptions), mixed into
one list of parlays.

How a prop's chance is judged:
  1. Our model (opportunity-based, injury boost for teammates, etc.)
  2. The market: the other sportsbooks' prices, cut removed, averaged
  3. Final chance = blend of the two (MARKET_WEIGHT)
  4. Edge = expected profit at BetMGM's actual price with that chance

Run:  python props_pipeline.py   (ODDS_API_KEY set in the terminal)
Odds are cached per week, so re-runs are free unless REFRESH_ODDS = True.
"""

from __future__ import annotations
import json
import os
import requests
import numpy as np
import pandas as pd

# =====================================================================
#  YOUR SETTINGS -- edit these, save, and run.
# =====================================================================
SEASON = 2026
WEEK = 4
BANKROLL = 2000

STAKE = 10               # dollars per parlay
MIN_LEGS = 4
MAX_LEGS = 6
MIN_PAYOUT = 1_000       # skip parlays paying less than this on STAKE
MAX_PAYOUT = 100_000     # skip parlays paying more than this on STAKE
SORT_BY = "hit_chance"   # "hit_chance" or "edge"
MIN_EDGE = 0.05          # each leg needs at least +5% expected profit at BetMGM's price
HOW_MANY = 15

INCLUDE = {
    "td":   True,   # Anytime TD
    "pass": True,   # Passing yards milestones
    "rush": True,   # Rushing yards milestones
    "rec":  True,   # Receiving yards milestones
    "recs": True,   # Receptions milestones
}

BET_BOOK = "betmgm"      # where you place bets
# Other books used only to read the market (up to 9 cost no extra API requests).
COMPARE_BOOKS = ["draftkings", "fanduel", "williamhill_us", "espnbet", "betrivers", "fanatics"]
MARKET_WEIGHT = 0.6      # 0 = trust only our model, 1 = trust only the other books
MIN_COMPARE_BOOKS = 2    # need at least this many other books pricing a prop to use the market

REFRESH_ODDS = False     # True = pull new prices (uses API requests)
# =====================================================================

from weekly_pipeline import build_current_week_matchups, TEAM_FULL_NAMES, SCHEDULE_URL, EVENTS_URL, EVENT_ODDS_URL_TMPL
from data_loader import load_pbp
from share_model import build_share_projections, implied_team_points
from yards_model import (STATS, DEFAULT_PARAMS, load_weekly_stats, opportunity_table, project_stat,
                         prob_at_least, threshold_from_point)
from features import load_snaps, snap_shares, opponent_factors
from calibrate import apply_correction
from devig_props import american_to_prob, american_to_decimal, ASSUMED_TOTAL_VIG
from parlay_builder import ATDLeg, rank_weekly_parlays
from name_match import normalize_full_name

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
MARKET_KEYS = {"td": "player_anytime_td", "pass": "player_pass_yds_alternate",
               "rush": "player_rush_yds_alternate", "rec": "player_reception_yds_alternate",
               "recs": "player_receptions_alternate"}
KEY_TO_STAT = {v: k for k, v in MARKET_KEYS.items()}
FULL_TO_ABBR = {v: k for k, v in TEAM_FULL_NAMES.items()}
MAX_PLAUSIBLE_EDGE = 1.0


# ---------------------------------------------------------------- odds
def fetch_odds(season, week, markets, books):
    path = f"odds_cache_{season}_wk{week:02d}.json"
    if os.path.exists(path) and not REFRESH_ODDS:
        cached = json.load(open(path))
        if set(markets) <= set(cached["markets"]) and set(books) <= set(cached.get("books", [])):
            print(f"  Using saved odds from {cached['pulled_at']} -- 0 API requests used. (REFRESH_ODDS = True for fresh.)")
            return cached["events"]
        print("  Saved odds don't cover your current prop types/books -- pulling fresh.")
    if not ODDS_API_KEY:
        raise RuntimeError("Set the ODDS_API_KEY environment variable first.")

    resp = requests.get(EVENTS_URL, params={"apiKey": ODDS_API_KEY})
    resp.raise_for_status()
    sched = pd.read_parquet(SCHEDULE_URL)
    wk = sched[(sched["season"] == season) & (sched["week"] == week)]
    start = pd.to_datetime(wk["gameday"]).min() - pd.Timedelta(days=1)
    end = pd.to_datetime(wk["gameday"]).max() + pd.Timedelta(days=2)
    now = pd.Timestamp.now(tz="UTC").tz_convert(None)
    events = [e for e in resp.json()
              if start <= pd.to_datetime(e["commence_time"], utc=True).tz_convert(None) <= end
              and pd.to_datetime(e["commence_time"], utc=True).tz_convert(None) > now]
    print(f"  {len(events)} upcoming week-{week} games. Cost: about {len(events) * len(markets)} API requests.")

    out, remaining = [], None
    for e in events:
        r = requests.get(EVENT_ODDS_URL_TMPL.format(event_id=e["id"]), params={
            "apiKey": ODDS_API_KEY, "regions": "us", "markets": ",".join(markets),
            "oddsFormat": "american", "bookmakers": ",".join(books)})
        remaining = r.headers.get("x-requests-remaining", remaining)
        if r.status_code != 200:
            print(f"  [warn] {e['away_team']} @ {e['home_team']}: {r.status_code} {r.text[:120]}")
            continue
        out.append(r.json())
    json.dump({"markets": markets, "books": books, "pulled_at": pd.Timestamp.now().strftime("%a %b %d %I:%M %p"),
               "events": out}, open(path, "w"))
    if remaining is not None:
        print(f"  API requests left this month: {remaining}")
    return out


def parse_odds(events):
    """One row per (book, player, stat, point) with the Yes/Over price and, if offered, the No/Under price."""
    rows = {}
    for ev in events:
        home, away = FULL_TO_ABBR.get(ev["home_team"]), FULL_TO_ABBR.get(ev["away_team"])
        for bm in ev.get("bookmakers", []):
            for mk in bm.get("markets", []):
                stat = KEY_TO_STAT.get(mk["key"])
                if stat is None:
                    continue
                for o in mk.get("outcomes", []):
                    player = o.get("description")
                    if not player:
                        continue
                    point = None if stat == "td" else o.get("point")
                    if stat != "td" and point is None:
                        continue
                    key = (bm["key"], player, stat, point, ev["id"])
                    row = rows.setdefault(key, {"book": bm["key"], "player_name": player, "stat": stat,
                                                "point": point, "game_id": ev["id"], "home": home, "away": away,
                                                "yes": None, "no": None})
                    if o.get("name") in ("Yes", "Over"):
                        row["yes"] = int(o["price"])
                    elif o.get("name") in ("No", "Under"):
                        row["no"] = int(o["price"])
    df = pd.DataFrame(rows.values())
    if df.empty:
        return df
    df["point"] = pd.to_numeric(df["point"]).fillna(-1.0)   # -1 = Anytime TD (no line)
    return df[df["yes"].notna()]


def fair_prob(yes, no):
    """Book's chance with its cut removed: exact if both sides are offered, else an assumed cut."""
    py = american_to_prob(int(yes))
    if no is not None and not pd.isna(no):
        return py / (py + american_to_prob(int(no)))
    return py / (1 + ASSUMED_TOTAL_VIG)


# ---------------------------------------------------------------- pipeline
def run(season=SEASON, week=WEEK):
    stats_on = [s for s, on in INCLUDE.items() if on]
    markets = [MARKET_KEYS[s] for s in stats_on]
    books = [BET_BOOK] + [b for b in COMPARE_BOOKS if b != BET_BOOK][:9]

    print("Loading schedule, rosters, injuries, snaps, and stats...")
    sched = pd.read_parquet(SCHEDULE_URL)
    matchups, team_lookup, full_name_lookup, out_team, pos_of = build_current_week_matchups(season, week, return_out=True)
    team_pts = implied_team_points(sched, season, week)
    try:
        snaps = snap_shares(load_snaps(season), week)
    except Exception:
        print("  [note] snap counts unavailable -- injury boost will split evenly.")
        snaps = {}

    td_prob = {}
    if "td" in stats_on:
        print("Projecting anytime-TD chances...")
        try:
            cal = json.load(open("calibration.json"))
        except FileNotFoundError:
            cal = {"a": 0.0, "b": 1.0, "params": {}}
        proj = build_share_projections(load_pbp(season), load_pbp(season - 1), sched, season, week, team_lookup,
                                       full_name_lookup, params=cal.get("params"), out_team=out_team,
                                       snaps=snaps, pos_of=pos_of)
        proj["model_prob"] = apply_correction(proj["model_prob"], cal["a"], cal["b"])
        td_prob = dict(zip(proj["player_id"], proj["model_prob"]))

    yard_mean, yard_params = {}, {}
    if any(s in stats_on for s in STATS):
        print("Projecting yards and receptions (opportunities x efficiency)...")
        try:
            ycal = json.load(open("calibration_yards.json"))
        except FileNotFoundError:
            print("  [note] calibration_yards.json not found -- using defaults. Run calibrate_yards.py.")
            ycal = DEFAULT_PARAMS
        cur, prior = load_weekly_stats(season), load_weekly_stats(season - 1)
        team_all = {**team_lookup, **out_team}
        for s in STATS:
            if s not in stats_on:
                continue
            par = {**DEFAULT_PARAMS[s], **ycal.get(s, {})}
            table = opportunity_table(cur, prior, week, s, par["decay"])
            pr = project_stat(table, cur, week, s, par, team_all, team_pts, matchups,
                              out=set(out_team), snaps=snaps)
            yard_mean[s], yard_params[s] = dict(zip(pr["player_id"], pr["mean"])), par

    print("Getting odds (BetMGM + comparison books)...")
    odds = parse_odds(fetch_odds(season, week, markets, books))
    if odds.empty:
        print("No odds came back. If this keeps happening, run debug_odds.py and send me the output.")
        return [], []

    # Who is each odds row about? Must be a roster player on one of the two teams in that game.
    name_to_pid = {(normalize_full_name(n), team_lookup[p]): p for p, n in full_name_lookup.items() if p in team_lookup}
    def find(row):
        k = normalize_full_name(row.player_name)
        for t in (row.home, row.away):
            if (k, t) in name_to_pid:
                return name_to_pid[(k, t)], t
        return None, None
    odds[["player_id", "team"]] = odds.apply(lambda r: pd.Series(find(r)), axis=1)
    odds = odds[odds["player_id"].notna()]
    odds["fair"] = [fair_prob(y, n) for y, n in zip(odds["yes"], odds["no"])]

    # Market consensus from the OTHER books, per player/prop/line
    others = odds[odds["book"] != BET_BOOK]
    cons = others.groupby(["player_id", "stat", "point"], dropna=False).agg(
        consensus=("fair", "median"), n_books=("book", "nunique")).reset_index()
    mgm = odds[odds["book"] == BET_BOOK].merge(cons, on=["player_id", "stat", "point"], how="left")
    if mgm.empty:
        print(f"No {BET_BOOK} prices found for this week yet.")
        return [], []

    legs, n_market = [], 0
    for r in mgm.itertuples():
        pid, team = r.player_id, r.team
        if r.stat == "td":
            model, prop = td_prob.get(pid), "Anytime TD"
        else:
            mean = yard_mean.get(r.stat, {}).get(pid)
            thr = threshold_from_point(float(r.point))
            prop = f"{thr}+ {STATS[r.stat][4]}"
            model = None if mean is None else float(prob_at_least(mean, thr, yard_params[r.stat]["shape"]))
        use_market = pd.notna(r.consensus) and r.n_books >= MIN_COMPARE_BOOKS
        if model is None and not use_market:
            continue
        if model is None:
            final = r.consensus            # no model view: market only
        elif use_market:
            final = MARKET_WEIGHT * r.consensus + (1 - MARKET_WEIGHT) * model
            n_market += 1
        else:
            final = model
        dec = american_to_decimal(int(r.yes))
        legs.append(ATDLeg(
            player_name=full_name_lookup.get(pid, r.player_name), player_id=pid, team=team,
            opponent=matchups[team], game_id=r.game_id, model_prob=float(final),
            market_prob=1 / dec,              # BetMGM's actual payout -> edge = expected profit
            yes_odds_american=int(r.yes), yes_odds_decimal=dec, book=BET_BOOK, prop=prop,
            our_model=model, consensus=(float(r.consensus) if use_market else None),
            n_books=(int(r.n_books) if use_market else 0)))
    print(f"  {len(legs)} BetMGM props priced ({n_market} compared against other books).")

    best = {}
    for l in legs:
        k = (l.player_id, l.prop)
        if k not in best or l.edge_pct > best[k].edge_pct:
            best[k] = l
    legs = list(best.values())

    suspect = [l for l in legs if l.edge_pct > MAX_PLAUSIBLE_EDGE]
    legs = [l for l in legs if l.edge_pct <= MAX_PLAUSIBLE_EDGE]
    if suspect:
        print(f"\n  [suspect] {len(suspect)} props set aside (looks too good to be true) -- check by hand:")
        for l in sorted(suspect, key=lambda l: l.edge_pct, reverse=True)[:10]:
            print(f"    {l.player_name:22s} {l.prop:16s} {l.yes_odds_american:+d}  {l.summary()}")

    good = [l for l in legs if l.edge_pct > MIN_EDGE]
    print(f"\n{len(good)} props clear the +{MIN_EDGE:.0%} edge bar "
          f"({sum('TD' in l.prop for l in good)} TD, {sum('Yds' in l.prop for l in good)} yards, "
          f"{sum('Receptions' in l.prop for l in good)} receptions).")
    print("  (BetMGM = what its price implies; Mkt = other books' average; Model = ours; Final = blend)\n")
    for l in sorted(good, key=lambda l: l.edge_pct, reverse=True)[:15]:
        print(f"  {l.player_name:22s} {l.prop:16s} {l.team}-{l.opponent:4s} {l.yes_odds_american:+5d}  "
              f"{l.summary()}  edge={l.edge_pct:+.0%}")

    parlays = rank_weekly_parlays(legs, BANKROLL, min_legs=MIN_LEGS, max_legs=MAX_LEGS, top_n=HOW_MANY,
                                  min_edge_pct=MIN_EDGE, stake=STAKE, min_payout=MIN_PAYOUT,
                                  max_payout=MAX_PAYOUT, sort_by=SORT_BY)
    if not parlays:
        print(f"\nNo parlays with {MIN_LEGS}-{MAX_LEGS} legs paying ${MIN_PAYOUT:,}-${MAX_PAYOUT:,} on ${STAKE}.")
        print("Try more legs, a wider payout range, or a lower MIN_EDGE -- or there's little value this week.")
    else:
        print(f"\nTop {len(parlays)} parlays (all BetMGM):\n")
        for i, p in enumerate(parlays, 1):
            print(f"#{i}")
            print(p.describe())
            print()
    return legs, parlays


if __name__ == "__main__":
    run()
