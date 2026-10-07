"""
weekly_pipeline.py — Run this once per NFL week (Tue/Wed, after injury
reports firm up but before lines move much closer to kickoff) to get
your ranked ATD parlay shortlist for the week.

Pipeline:
  1. Pull play-by-play through the most recently completed week (free, nflverse)
  2. Compute red-zone usage + league baselines + opponent defense factors
  3. Project every player's anytime-TD probability for this week's matchups
  4. Pull this week's ATD odds from FanDuel/BetMGM (The Odds API -- the one
     paid API in this pipeline, same key you're already using for Sharp Futures)
  5. Devig, compute edge, rank individual legs
  6. Build and rank cross-game parlay combos

Fill in ODDS_API_KEY and the two TODOs marked below before running for real.
"""

from __future__ import annotations
import os
import json
import requests
import pandas as pd

from data_loader import load_pbp, compute_player_red_zone_usage, league_baseline_rates, team_red_zone_defense
from share_model import build_share_projections
from features import load_snaps, snap_shares
from devig_props import PropPrice, devig_two_way, american_to_decimal
from parlay_builder import ATDLeg, rank_weekly_parlays
from name_match import normalize_full_name
from calibrate import apply_correction


# =====================================================================
#  YOUR SETTINGS -- edit these, save, and run. Nothing else needs changing.
# =====================================================================
SEASON = 2026
WEEK = 4                 # the upcoming week you're betting on
BANKROLL = 2000

STAKE = 10               # dollars per parlay (your $1-20 range)
MIN_LEGS = 4             # fewest legs in a parlay
MAX_LEGS = 6             # most legs in a parlay
MIN_PAYOUT = 1_000       # skip parlays that pay less than this on STAKE
MAX_PAYOUT = 100_000     # skip parlays that pay more than this on STAKE
SORT_BY = "hit_chance"   # "hit_chance" = most likely to hit first, "edge" = biggest edge first
MIN_EDGE = 0.05          # a leg needs the model at least 5% above the book to be used
HOW_MANY = 15            # how many parlays to show
# =====================================================================

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"


EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
EVENT_ODDS_URL_TMPL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"
BOOKS = ["betmgm"]   # sportsbooks to use, e.g. ["fanduel", "betmgm"]

# nflverse team abbreviation -> The Odds API's full team name. Used to
# verify that an odds quote actually came from a game the player's own
# team is playing in (a hard check against name-matching mistakes).
TEAM_FULL_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LA": "Los Angeles Rams", "LAC": "Los Angeles Chargers",
    "LV": "Las Vegas Raiders", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SEA": "Seattle Seahawks", "SF": "San Francisco 49ers", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}


def fetch_atd_odds(season: int, week: int) -> pd.DataFrame:
    """
    Pulls anytime-TD-scorer odds from FanDuel and BetMGM via The Odds API.
    Two-step process, since player props aren't in the bulk /odds endpoint:
      1. GET /events -- list of this week's game IDs
      2. GET /events/{id}/odds?markets=player_anytime_td -- per game

    Cost: 1 request for the event list + 1 request per game per call
    (~16/week here). Check your plan's monthly cap before running this
    across a full season -- this is the one place API usage adds up.

    Returns columns: player_name, team, game_id, book, yes_odds, no_odds
    """
    if not ODDS_API_KEY:
        raise RuntimeError("Set the ODDS_API_KEY environment variable first.")

    events_resp = requests.get(EVENTS_URL, params={"apiKey": ODDS_API_KEY})
    events_resp.raise_for_status()
    events = events_resp.json()

    # The events list includes every upcoming game, not just this week's.
    # Keep only games inside this week's date window (from the nflverse
    # schedule), so next week's prices never get mixed in and you don't
    # burn API requests on games you aren't betting yet.
    sched = pd.read_parquet(SCHEDULE_URL)
    wk = sched[(sched["season"] == season) & (sched["week"] == week)]
    start = pd.to_datetime(wk["gameday"]).min() - pd.Timedelta(days=1)
    end = pd.to_datetime(wk["gameday"]).max() + pd.Timedelta(days=2)
    events = [
        e for e in events
        if start <= pd.to_datetime(e["commence_time"], utc=True).tz_convert(None) <= end
    ]
    print(f"  {len(events)} games found for week {week}.")

    rows = []
    for event in events:
        event_id = event["id"]
        home_team, away_team = event["home_team"], event["away_team"]

        resp = requests.get(
            EVENT_ODDS_URL_TMPL.format(event_id=event_id),
            params={
                "apiKey": ODDS_API_KEY,
                "regions": "us",
                "markets": "player_anytime_td",
                "oddsFormat": "american",
                "bookmakers": ",".join(BOOKS),
            },
        )
        if resp.status_code != 200:
            print(f"  [warn] odds fetch failed for {away_team} @ {home_team}: {resp.status_code} {resp.text[:150]}")
            continue
        data = resp.json()

        game_id = f"{season}_{week:02d}_{away_team}_{home_team}"  # matches nflverse game_id format

        for bookmaker in data.get("bookmakers", []):
            book_key = bookmaker["key"]
            if book_key not in BOOKS:
                continue
            for market in bookmaker.get("markets", []):
                if market["key"] != "player_anytime_td":
                    continue
                # Outcomes come as one row per player with name='Yes'/'No' and description=player name
                # (confirm this shape against a live response -- prop market schemas do shift occasionally).
                by_player: dict[str, dict] = {}
                for outcome in market["outcomes"]:
                    player = outcome.get("description", outcome.get("name"))
                    by_player.setdefault(player, {})[outcome["name"]] = outcome["price"]

                for player_name, sides in by_player.items():
                    if "Yes" not in sides:
                        continue
                    rows.append({
                        "player_name": player_name,
                        "team": None,  # filled in later via roster join in run_weekly_pipeline
                        "game_id": game_id,
                        "event_home": home_team,
                        "event_away": away_team,
                        "commence_time": event.get("commence_time"),
                        "book": book_key,
                        "yes_odds": sides["Yes"],
                        "no_odds": sides.get("No"),
                    })

    return pd.DataFrame(rows)


SCHEDULE_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"
ROSTER_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/weekly_rosters/roster_weekly_{season}.parquet"
INJURIES_URL_TMPL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"


def build_current_week_matchups(season: int, week: int, return_out: bool = False):
    """
    Returns (matchups, team_lookup, full_name_lookup) for the given week:
      matchups: {team_abbrev: opponent_abbrev}
      team_lookup: {player_id: team_abbrev} -- current roster mapping
      full_name_lookup: {player_id: full_name} -- e.g. "Kyren Williams"

    full_name_lookup matters a lot: nflverse play-by-play only has
    abbreviated names ("K.Williams"), which collide across different real
    players (Kyren Williams, Kyle Williams, Ke'Shawn Williams all reduce
    to the same "K.Williams" / "k.williams" key). The roster file has
    real full names, so projections should carry THOSE for matching
    against the Odds API's full names, instead of matching on the lossy
    abbreviated form.

    Both pulled fresh from nflverse's public data (free, no API key) --
    schedule from the 'schedules' release (games.parquet), rosters from
    the 'weekly_rosters' release, filtered to the most recent week with
    a roster snapshot so mid-week trades/signings are reflected.
    """
    games = pd.read_parquet(SCHEDULE_URL)
    week_games = games[(games["season"] == season) & (games["week"] == week)]
    if week_games.empty:
        raise ValueError(f"No scheduled games found for season {season} week {week} -- check the week number.")

    matchups = {}
    for _, g in week_games.iterrows():
        matchups[g["home_team"]] = g["away_team"]
        matchups[g["away_team"]] = g["home_team"]

    rosters = pd.read_parquet(ROSTER_URL_TMPL.format(season=season))
    season_rosters = rosters[rosters["season"] == season]
    # Use the latest available roster week at or before the target week,
    # since the target week's roster file may not be posted until game day.
    available_weeks = season_rosters[season_rosters["week"] <= week]["week"]
    if available_weeks.empty:
        raise ValueError(f"No roster data available yet for season {season} at or before week {week}.")
    latest_week = available_weeks.max()
    current_rosters = season_rosters[
        (season_rosters["week"] == latest_week) & (season_rosters["status"] == "ACT")
    ]

    team_lookup = dict(zip(current_rosters["gsis_id"], current_rosters["team"]))
    full_name_lookup = dict(zip(current_rosters["gsis_id"], current_rosters["full_name"]))

    out_team = {}  # injured players removed below -> their team (used for the teammate boost)
    pos_lookup = dict(zip(current_rosters["gsis_id"], current_rosters["position"]))

    # Drop anyone listed Out or Doubtful on this week's official injury
    # report (free nflverse data). Removing them from team_lookup means no
    # projection is made for them, so they can never land in a parlay.
    try:
        inj = pd.read_parquet(INJURIES_URL_TMPL.format(season=season))
        inj = inj[(inj["season"] == season) & (inj["week"] == week)]
        if inj.empty:
            print(f"  [note] week {week} injury report not posted yet -- re-run later in the week "
                  f"(final statuses usually come out Friday).")
        else:
            out = inj[inj["report_status"].isin(["Out", "Doubtful"])]
            removed = [pid for pid in out["gsis_id"] if pid in team_lookup]
            for pid in removed:
                out_team[pid] = team_lookup.pop(pid)
            print(f"  Injury report: removed {len(removed)} players listed Out or Doubtful.")
    except Exception as e:
        print(f"  [note] couldn't load the injury report ({str(e)[:60]}) -- continuing without it.")

    if return_out:
        return matchups, team_lookup, full_name_lookup, out_team, pos_lookup
    return matchups, team_lookup, full_name_lookup


def run_weekly_pipeline(season: int, week: int, bankroll: float, min_edge_pct: float = 0.10):
    print(f"Loading play-by-play through week {week - 1}...")
    pbp = load_pbp(season)
    print(f"Loading {season - 1} season for established-role blending...")
    pbp_prior = load_pbp(season - 1)
    sched = pd.read_parquet(SCHEDULE_URL)

    print("Building matchups and rosters for this week...")
    matchups, team_lookup, full_name_lookup, out_team, pos_of = build_current_week_matchups(season, week, return_out=True)

    print("Projecting anytime-TD probabilities (team expected TDs x player share)...")
    try:
        params = json.load(open("calibration.json")).get("params")
    except FileNotFoundError:
        params = None
    try:
        snaps = snap_shares(load_snaps(season), week)
    except Exception:
        snaps = {}
    projections = build_share_projections(
        pbp, pbp_prior, sched, season, week, team_lookup, full_name_lookup=full_name_lookup,
        params=params, out_team=out_team, snaps=snaps, pos_of=pos_of,
    )
    wk = sched[(sched["season"] == season) & (sched["week"] == week)]
    if wk["total_line"].isna().any():
        print(f"  [note] {int(wk['total_line'].isna().sum())} games have no betting line yet -- "
              f"those teams use a league-average 22 points. Re-run later in the week for better numbers.")

    # Apply the calibration correction from calibrate.py, if it's been run.
    try:
        with open("calibration.json") as f:
            cal = json.load(f)
        projections["model_prob"] = apply_correction(projections["model_prob"], cal["a"], cal["b"])
        print(f"  Applied calibration correction (fit on {cal['fit_on']}).")
    except FileNotFoundError:
        print("  [note] calibration.json not found -- run calibrate.py once to correct the model's percentages.")

    print("Fetching current ATD odds from FanDuel/BetMGM...")
    odds_df = fetch_atd_odds(season, week)

    # Primary match: strict full-name key. Projections now carry real full
    # names from the roster file (see full_name_lookup in
    # build_current_week_matchups), matching the Odds API's full names --
    # this is what fixed a real bug found in testing, where the old
    # initial+lastname shortcut collapsed Kyren Williams, Kyle Williams,
    # and Ke'Shawn Williams into one incorrectly-shared bucket.
    projections = projections.copy()
    odds_df = odds_df.copy()
    projections["name_key"] = projections["player_name"].apply(normalize_full_name)
    odds_df["name_key"] = odds_df["player_name"].apply(normalize_full_name)

    merged = pd.merge(
        projections, odds_df, on="name_key", how="inner", suffixes=("", "_odds")
    )

    # NOTE: an earlier version of this pipeline had a "loose key" fallback
    # for odds rows that didn't strictly match a projection by full name.
    # It was removed after a real bug found in testing: a low-usage player
    # (Ke'Shawn Williams) gets correctly filtered out of projections for
    # having too little data to trust, but his real odds still exist in
    # the market feed. His loose key ("k.williams") happened to collide
    # with an unrelated, legitimate player (Kyren Williams) who IS in
    # projections -- the fallback silently paired Ke'Shawn's real odds
    # onto Kyren's model row. There's no way to verify from names alone
    # that a loose-key match is actually the same real person, so rather
    # than risk that again, unmatched odds rows are just left unmatched.
    # A missed player is a far better failure mode than a wrong pairing
    # when real money is on the line.
    unmatched_count = (~odds_df["name_key"].isin(projections["name_key"])).sum()
    if unmatched_count > 0:
        print(f"  [note] {unmatched_count} odds rows didn't match a model projection by exact "
              f"full name and were skipped (most likely low-usage players my model already "
              f"filters out for having too little data to trust -- not a concern).")

    print(f"  Matched {len(merged)} of {len(odds_df)} odds rows to model projections by name.")

    # Hard check: the odds quote must come from a game the player's own
    # team is actually playing in. A same-name player on another team
    # (or any other mispairing) gets dropped here no matter how the
    # names happened to line up.
    merged["team_full"] = merged["team"].map(TEAM_FULL_NAMES)
    team_ok = (merged["team_full"] == merged["event_home"]) | (merged["team_full"] == merged["event_away"])
    if (~team_ok).sum() > 0:
        print(f"  [note] dropped {(~team_ok).sum()} odds rows whose game didn't involve the player's team.")
    merged = merged[team_ok]

    legs = []
    for _, row in merged.iterrows():
        price = PropPrice(row["player_name"], yes_odds=row["yes_odds"], no_odds=row.get("no_odds"))
        market_prob = devig_two_way(price)
        legs.append(ATDLeg(
            player_name=row["player_name"],
            player_id=row["player_id"],
            team=row["team"],
            opponent=row["opponent"],
            game_id=row["game_id"],
            model_prob=row["model_prob"],
            market_prob=market_prob,
            yes_odds_american=row["yes_odds"],
            yes_odds_decimal=american_to_decimal(row["yes_odds"]),
            book=row["book"],
        ))

    # Multiple sportsbooks (or an imperfect name match) can produce more than
    # one quote for the same underlying player -- keep only the single best
    # (highest-edge) quote per player before counting/ranking anything, so
    # one real player never shows up as 2-3 "different" legs.
    best_by_player: dict[str, ATDLeg] = {}
    for leg in legs:
        current_best = best_by_player.get(leg.player_id)
        if current_best is None or leg.edge_pct > current_best.edge_pct:
            best_by_player[leg.player_id] = leg
    legs = list(best_by_player.values())

    # Sanity check. Real edges on anytime-TD props are small -- a few
    # percent. If the model claims a player is more than twice as likely
    # to score as the market thinks, that is almost always bad data or a
    # blind spot in the model (injury, role change it can't see), not a
    # gift. Set those aside for you to look at by hand instead of letting
    # them drive the parlays.
    MAX_PLAUSIBLE_EDGE = 1.0  # i.e. model prob <= 2x market prob
    suspect = [l for l in legs if l.edge_pct > MAX_PLAUSIBLE_EDGE]
    legs = [l for l in legs if l.edge_pct <= MAX_PLAUSIBLE_EDGE]
    if suspect:
        print(f"\n  [suspect] {len(suspect)} legs set aside -- model is more than 2x the market, "
              f"check these by hand (injury? role change?) before trusting them:")
        for l in sorted(suspect, key=lambda l: l.edge_pct, reverse=True)[:10]:
            print(f"    {l.player_name:20s} {l.team}  model={l.model_prob:.1%}  "
                  f"market={l.market_prob:.1%}  odds={l.yes_odds_american:+d} ({l.book})")

    print(f"\n{sum(1 for l in legs if l.edge_pct > min_edge_pct)} legs clear the {min_edge_pct:.0%} edge threshold "
          f"(deduped to one quote per player, {len(legs)} unique players total).\n")
    print("Top individual legs by edge:")
    for leg in sorted(legs, key=lambda l: l.edge_pct, reverse=True)[:10]:
        print(f"  {leg.player_name:20s} {leg.team}-{leg.opponent:4s} odds={leg.yes_odds_american:+d}  model={leg.model_prob:.1%}  market={leg.market_prob:.1%}  edge={leg.edge_pct:+.1%}  ({leg.book})")

    parlays = rank_weekly_parlays(
        legs, bankroll, min_legs=MIN_LEGS, max_legs=MAX_LEGS, top_n=HOW_MANY,
        min_edge_pct=min_edge_pct, stake=STAKE, min_payout=MIN_PAYOUT,
        max_payout=MAX_PAYOUT, sort_by=SORT_BY,
    )
    if not parlays:
        print(f"\nNo parlays found with {MIN_LEGS}-{MAX_LEGS} legs paying ${MIN_PAYOUT:,}-${MAX_PAYOUT:,} on a ${STAKE} bet.")
        print("Try: more legs, a wider payout range, or a lower MIN_EDGE. (Few results can also just mean")
        print("the model doesn't see much value this week -- that's a legitimate answer.)")

    print(f"\nTop {len(parlays)} ranked parlays:\n")
    for i, p in enumerate(parlays, 1):
        print(f"#{i}")
        print(p.describe())
        print()

    return legs, parlays


if __name__ == "__main__":
    # Settings are at the top of this file.
    run_weekly_pipeline(season=SEASON, week=WEEK, bankroll=BANKROLL, min_edge_pct=MIN_EDGE)
