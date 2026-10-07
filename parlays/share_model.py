"""
share_model.py -- The improved touchdown model.

Player's expected TDs = (team's expected TDs this week) x (player's share
of his team's scoring chances)

  1. TEAM EXPECTED TDs: from the game's spread and over/under (free, in
     the nflverse schedule). E.g. total 51.5, home favored by 2.5 ->
     home expected ~27 points -> about 3 offensive touchdowns.

  2. PLAYER SHARE: every carry and target is worth a different amount
     depending on where it happens. A carry at the 1-yard line scores
     ~45% of the time; a carry at midfield almost never does. We add up
     the "touchdown value" of every touch a player got, and divide by
     his team's total in the same games. That's his share.

  3. Share is blended with last season's share (counted as 6 games' worth)
     so a few lucky plays early in the year don't swing it too much.

Then: chance of at least one TD = 1 - e^(-expected TDs).
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from features import redistribute

# Yard-line zones (distance from the end zone). Closer = far more valuable.
ZONE_BINS = [0, 2, 5, 10, 20, 100]
ZONE_LABELS = ["1-2", "3-5", "6-10", "11-20", "21+"]

PRIOR_WEIGHT_GAMES = 6.0        # last season counts like this many games
NO_PRIOR_WEIGHT_GAMES = 2.0     # rookies/newcomers: pull toward zero share this hard
LEAGUE_AVG_POINTS = 22.0        # fallback if a game has no betting line


def _reg_only(pbp: pd.DataFrame) -> pd.DataFrame:
    if "season_type" in pbp.columns:
        return pbp[pbp["season_type"] == "REG"]
    return pbp


def touch_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per carry or target: who, which team/game, which zone, did it score."""
    df = _reg_only(pbp)
    df = df[df["yardline_100"].notna()]
    rush = df[(df["rush_attempt"] == 1) & df["rusher_player_id"].notna()].assign(
        kind="rush", player_id=lambda d: d["rusher_player_id"], td=lambda d: d["rush_touchdown"])
    tgt = df[(df["pass_attempt"] == 1) & df["receiver_player_id"].notna()].assign(
        kind="target", player_id=lambda d: d["receiver_player_id"], td=lambda d: d["pass_touchdown"])
    t = pd.concat([rush, tgt], ignore_index=True)
    t["zone"] = pd.cut(t["yardline_100"], bins=ZONE_BINS, labels=ZONE_LABELS, include_lowest=True)
    return t[["season", "week", "game_id", "posteam", "player_id", "kind", "zone", "td"]]


def zone_td_rates(touches: pd.DataFrame) -> dict:
    """League TD rate for each (carry/target, zone) combo. Built from a full past season."""
    r = touches.groupby(["kind", "zone"], observed=True)["td"].mean()
    return {(k, z): float(v) for (k, z), v in r.items()}


def add_touch_value(touches: pd.DataFrame, rates: dict) -> pd.DataFrame:
    t = touches.copy()
    t["xtd"] = [rates.get((k, z), 0.0) for k, z in zip(t["kind"], t["zone"])]
    return t


def player_shares(touches: pd.DataFrame, target_week: int | None = None, decay: float = 1.0) -> pd.DataFrame:
    """
    Per player: his touch value / his team's touch value, counting only
    games he played in. With decay < 1, recent games count more.
    """
    pg = touches.groupby(["game_id", "week", "posteam", "player_id"], as_index=False)["xtd"].sum()
    tg = touches.groupby(["game_id", "posteam"], as_index=False)["xtd"].sum().rename(columns={"xtd": "team_xtd"})
    pg = pg.merge(tg, on=["game_id", "posteam"])
    pg["w"] = 1.0 if (target_week is None or decay == 1.0) else decay ** (target_week - 1 - pg["week"])
    pg["wx"], pg["wt"] = pg["w"] * pg["xtd"], pg["w"] * pg["team_xtd"]
    out = pg.groupby("player_id").agg(
        player_xtd=("wx", "sum"), team_xtd=("wt", "sum"), games=("game_id", "nunique"), wgames=("w", "sum"),
    ).reset_index()
    out["share"] = out["player_xtd"] / out["team_xtd"]
    return out


def implied_team_points(sched: pd.DataFrame, season: int, week: int) -> dict:
    """{team: expected points this week} from spread + over/under."""
    g = sched[(sched["season"] == season) & (sched["week"] == week)]
    pts = {}
    for _, r in g.iterrows():
        if pd.notna(r["spread_line"]) and pd.notna(r["total_line"]):
            home = r["total_line"] / 2 + r["spread_line"] / 2   # positive spread = home favored
            away = r["total_line"] / 2 - r["spread_line"] / 2
        else:
            home = away = LEAGUE_AVG_POINTS
        pts[r["home_team"]] = home
        pts[r["away_team"]] = away
    return pts


def fit_points_to_tds(pbp: pd.DataFrame, sched: pd.DataFrame, season: int) -> tuple[float, float]:
    """
    How many rushing+receiving TDs a team actually scores for a given
    expected-points number, fit on a completed season: tds = c + k * points.
    """
    df = _reg_only(pbp)
    tds = df.groupby(["game_id", "posteam"]).apply(
        lambda d: d["rush_touchdown"].sum() + d["pass_touchdown"].sum(), include_groups=False
    ).reset_index(name="tds")
    s = sched[(sched["season"] == season) & (sched["game_type"] == "REG")].dropna(subset=["spread_line", "total_line"])
    rows = []
    for _, r in s.iterrows():
        rows.append((r["game_id"], r["home_team"], r["total_line"] / 2 + r["spread_line"] / 2))
        rows.append((r["game_id"], r["away_team"], r["total_line"] / 2 - r["spread_line"] / 2))
    pts = pd.DataFrame(rows, columns=["game_id", "posteam", "points"])
    d = tds.merge(pts, on=["game_id", "posteam"])
    k, c = np.polyfit(d["points"], d["tds"], 1)
    return float(c), float(k)


DEFAULT_TD_PARAMS = {"decay": 1.0, "inj": 0.0, "opp": 0.0}


def share_components(cur_pbp, prior_pbp, sched, season, week, team_lookup, decay=1.0,
                     out_team=None, snaps=None, pos_of=None, opp_td=None, min_games_played=2):
    """
    Every ingredient of the TD projection, kept separate so the backtest
    can test each feature's strength:
      base_share   blended share of team scoring chances (recency via decay)
      inj_add      extra share he'd get if ALL of an injured same-position
                   teammate's share went to him and his position-mates
      team_exp_tds team's expected rushing+receiving TDs from the betting line
      opp_raw      how many TDs this opponent allows to his position (1.0 = avg)
    """
    out_team, snaps, pos_of, opp_td = out_team or {}, snaps or {}, pos_of or {}, opp_td or {}
    prior_touches = touch_table(prior_pbp)
    rates = zone_td_rates(prior_touches)
    c, k = fit_points_to_tds(prior_pbp, sched, season - 1)

    cur = touch_table(cur_pbp)
    cur = cur[cur["week"] < week]
    cur_sh = player_shares(add_touch_value(cur, rates), week, decay).set_index("player_id")
    prior_sh = player_shares(add_touch_value(prior_touches, rates)).set_index("player_id")

    team_pts = implied_team_points(sched, season, week)
    wk = sched[(sched["season"] == season) & (sched["week"] == week)]
    opp = {**dict(zip(wk["home_team"], wk["away_team"])), **dict(zip(wk["away_team"], wk["home_team"]))}

    shares, games = {}, {}
    everyone = {**team_lookup, **out_team}
    for pid in set(cur_sh.index) | set(prior_sh.index):
        if pid not in everyone:
            continue
        g = cur_sh.at[pid, "games"] if pid in cur_sh.index else 0
        wg = cur_sh.at[pid, "wgames"] if pid in cur_sh.index else 0.0
        cs = cur_sh.at[pid, "share"] if pid in cur_sh.index else 0.0
        if pid in prior_sh.index:
            sh = (wg * cs + PRIOR_WEIGHT_GAMES * prior_sh.at[pid, "share"]) / (wg + PRIOR_WEIGHT_GAMES)
        else:
            sh = (wg * cs) / (wg + NO_PRIOR_WEIGHT_GAMES)
        shares[pid], games[pid] = sh, g

    base = {p: v for p, v in shares.items() if p in team_lookup}
    full = redistribute({**base, **{p: shares[p] for p in out_team if p in shares}},
                        everyone, pos_of, set(out_team), snaps, 1.0)

    rows = []
    for pid, sh in base.items():
        team = team_lookup[pid]
        if games.get(pid, 0) < min_games_played or team not in team_pts:
            continue
        rows.append({"player_id": pid, "team": team, "opponent": opp[team], "position": pos_of.get(pid),
                     "team_implied_pts": round(team_pts[team], 1),
                     "team_exp_tds": max(c + k * team_pts[team], 0.3),
                     "base_share": sh, "inj_add": full.get(pid, sh) - sh,
                     "opp_raw": opp_td.get((opp[team], pos_of.get(pid)), 1.0)})
    return pd.DataFrame(rows)


def lambda_from_components(d: pd.DataFrame, p: dict) -> np.ndarray:
    share = d["base_share"].values + p["inj"] * d["inj_add"].values
    return d["team_exp_tds"].values * share * d["opp_raw"].values ** p["opp"]


def build_share_projections(cur_pbp, prior_pbp, sched, season, week, team_lookup,
                            full_name_lookup=None, min_games_played=2, params=None,
                            out_team=None, snaps=None, pos_of=None, opp_td=None):
    """Anytime-TD chance for every active player this week."""
    p = {**DEFAULT_TD_PARAMS, **(params or {})}
    d = share_components(cur_pbp, prior_pbp, sched, season, week, team_lookup, p["decay"],
                         out_team, snaps, pos_of, opp_td, min_games_played)
    if d.empty:
        return d
    full_name_lookup = full_name_lookup or {}
    d["player_name"] = d["player_id"].map(lambda x: full_name_lookup.get(x, x))
    d["share"] = d["base_share"] + p["inj"] * d["inj_add"]
    d["lambda_total"] = lambda_from_components(d, p)
    d["model_prob"] = 1 - np.exp(-d["lambda_total"])
    return d.sort_values("model_prob", ascending=False).reset_index(drop=True)
