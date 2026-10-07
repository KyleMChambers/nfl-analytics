"""
data_loader.py — Pulls free, real NFL play-by-play data from nflverse
(no API key, no paid data) and turns it into the two things the TD
model needs per player:

  1. Red-zone opportunity volume (RZ carries/game, RZ targets/game)
  2. Historical TD conversion rate per RZ touch, for shrinkage baselines

nflverse publishes full play-by-play as a public GitHub release, updated
after every game — this is the same free-data approach already used for
the NFL Elo model, so no new data source or API to maintain.
"""

from __future__ import annotations
import pandas as pd
import numpy as np

PBP_COLUMNS = [
    "season_type",
    "season", "week", "game_id", "posteam", "defteam", "play_type",
    "yardline_100", "rush_attempt", "pass_attempt", "complete_pass",
    "touchdown", "rush_touchdown", "pass_touchdown", "two_point_attempt",
    "goal_to_go", "rusher_player_name", "rusher_player_id",
    "receiver_player_name", "receiver_player_id",
]

RED_ZONE_YARDLINE = 20  # yardline_100 <= 20 means inside the opponent's 20


def load_pbp(season: int) -> pd.DataFrame:
    """Pull one season of play-by-play from nflverse's public data release."""
    url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
    df = pd.read_parquet(url, columns=PBP_COLUMNS)
    return df


def load_multi_season_pbp(seasons: list[int]) -> pd.DataFrame:
    frames = [load_pbp(s) for s in seasons]
    return pd.concat(frames, ignore_index=True)


def compute_player_red_zone_usage(pbp: pd.DataFrame, through_week: int | None = None) -> pd.DataFrame:
    """
    Per player, per season: how many red-zone carries/targets they got,
    how many of those went for TDs, and how many games they've played
    (for a per-game rate). Restrict to <= through_week to avoid leaking
    future weeks into "this week's" input when backtesting.
    """
    df = pbp.copy()
    if through_week is not None:
        df = df[df["week"] <= through_week]

    rz = df[df["yardline_100"] <= RED_ZONE_YARDLINE]

    rushes = rz[rz["rush_attempt"] == 1].groupby(
        ["season", "rusher_player_id", "rusher_player_name"], dropna=True
    ).agg(
        rz_carries=("rush_attempt", "sum"),
        rz_rush_tds=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id", "rusher_player_name": "player_name"})

    targets = rz[rz["pass_attempt"] == 1].groupby(
        ["season", "receiver_player_id", "receiver_player_name"], dropna=True
    ).agg(
        rz_targets=("pass_attempt", "sum"),
        rz_rec_tds=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id", "receiver_player_name": "player_name"})

    # Games played, to get a per-game rate rather than a season total.
    games_rush = df[df["rush_attempt"] == 1].groupby(
        ["season", "rusher_player_id"]
    )["game_id"].nunique().reset_index(name="games_played").rename(columns={"rusher_player_id": "player_id"})
    games_rec = df[df["pass_attempt"] == 1].groupby(
        ["season", "receiver_player_id"]
    )["game_id"].nunique().reset_index(name="games_played").rename(columns={"receiver_player_id": "player_id"})
    games = pd.concat([games_rush, games_rec]).groupby(["season", "player_id"])["games_played"].max().reset_index()

    usage = pd.merge(rushes, targets, on=["season", "player_id", "player_name"], how="outer")
    usage = pd.merge(usage, games, on=["season", "player_id"], how="left")

    for col in ["rz_carries", "rz_rush_tds", "rz_targets", "rz_rec_tds"]:
        usage[col] = usage[col].fillna(0)
    usage["games_played"] = usage["games_played"].fillna(1).clip(lower=1)

    usage["rz_carries_per_game"] = usage["rz_carries"] / usage["games_played"]
    usage["rz_targets_per_game"] = usage["rz_targets"] / usage["games_played"]

    # Touches OUTSIDE the red zone, per player. Long touchdowns come from
    # these, and a star gets far more of them than a backup -- the model
    # used to assume a flat 2 per game for everyone, which flattened the
    # difference between stars and depth players.
    open_field = df[df["yardline_100"] > RED_ZONE_YARDLINE]
    or_rush = open_field[open_field["rush_attempt"] == 1].groupby(["season", "rusher_player_id"]).agg(
        or_carries=("rush_attempt", "sum"), or_rush_tds=("rush_touchdown", "sum"),
    ).reset_index().rename(columns={"rusher_player_id": "player_id"})
    or_rec = open_field[open_field["pass_attempt"] == 1].groupby(["season", "receiver_player_id"]).agg(
        or_targets=("pass_attempt", "sum"), or_rec_tds=("pass_touchdown", "sum"),
    ).reset_index().rename(columns={"receiver_player_id": "player_id"})
    usage = usage.merge(or_rush, on=["season", "player_id"], how="left").merge(or_rec, on=["season", "player_id"], how="left")
    for col in ["or_carries", "or_rush_tds", "or_targets", "or_rec_tds"]:
        usage[col] = usage[col].fillna(0)
    usage["or_carries_per_game"] = usage["or_carries"] / usage["games_played"]
    usage["or_targets_per_game"] = usage["or_targets"] / usage["games_played"]

    return usage


def league_baseline_rates(usage: pd.DataFrame) -> dict:
    """
    League-wide TD-per-red-zone-touch rates, used as the shrinkage target
    for players with limited red-zone volume so far. Split rush vs. pass
    since goal-line rushing converts at a meaningfully higher rate than
    a red-zone target does.
    """
    total_rz_carries = usage["rz_carries"].sum()
    total_rz_rush_tds = usage["rz_rush_tds"].sum()
    total_rz_targets = usage["rz_targets"].sum()
    total_rz_rec_tds = usage["rz_rec_tds"].sum()

    out = {
        "rush_td_rate": total_rz_rush_tds / total_rz_carries if total_rz_carries else 0.15,
        "rec_td_rate": total_rz_rec_tds / total_rz_targets if total_rz_targets else 0.20,
    }
    if "or_carries" in usage.columns:
        oc, ot = usage["or_carries"].sum(), usage["or_targets"].sum()
        out["or_rush_td_rate"] = usage["or_rush_tds"].sum() / oc if oc else 0.01
        out["or_rec_td_rate"] = usage["or_rec_tds"].sum() / ot if ot else 0.015
    return out


def team_red_zone_defense(pbp: pd.DataFrame, through_week: int | None = None) -> pd.DataFrame:
    """
    Per defense: how often opponents scored TDs on red-zone plays against
    them, relative to league average. Used as a multiplicative adjustment
    on a player's expected TD rate for their upcoming opponent.
    """
    df = pbp.copy()
    if through_week is not None:
        df = df[df["week"] <= through_week]

    rz = df[df["yardline_100"] <= RED_ZONE_YARDLINE]
    rz_plays = rz[(rz["rush_attempt"] == 1) | (rz["pass_attempt"] == 1)]

    def_stats = rz_plays.groupby("defteam").agg(
        rz_plays_faced=("touchdown", "count"),
        rz_tds_allowed=("touchdown", "sum"),
    ).reset_index()

    league_rate = def_stats["rz_tds_allowed"].sum() / def_stats["rz_plays_faced"].sum()
    def_stats["rz_td_rate_allowed"] = def_stats["rz_tds_allowed"] / def_stats["rz_plays_faced"]
    # Shrink defense rate toward league average with a pseudo-count of 40 plays,
    # since a single season's red-zone defense sample is still fairly small.
    k = 40
    def_stats["rz_td_rate_allowed_shrunk"] = (
        def_stats["rz_tds_allowed"] + k * league_rate
    ) / (def_stats["rz_plays_faced"] + k)
    def_stats["opponent_factor"] = def_stats["rz_td_rate_allowed_shrunk"] / league_rate

    return def_stats[["defteam", "rz_td_rate_allowed_shrunk", "opponent_factor"]]


if __name__ == "__main__":
    # Sanity check against real 2026 data pulled live from nflverse.
    pbp = load_pbp(2026)
    print(f"Loaded {len(pbp)} plays from the 2026 season, weeks {sorted(pbp['week'].unique())}")

    usage = compute_player_red_zone_usage(pbp)
    usage_sorted = usage.sort_values("rz_carries_per_game", ascending=False)
    print("\nTop 10 red-zone rushing volume so far this season:")
    print(usage_sorted[["player_name", "rz_carries_per_game", "rz_carries", "rz_rush_tds", "games_played"]].head(10).to_string(index=False))

    baselines = league_baseline_rates(usage)
    print("\nLeague baseline TD rates per red-zone touch:", baselines)

    def_rz = team_red_zone_defense(pbp)
    print("\nRed-zone defense factors (1.0 = league average, >1.0 = gives up more TDs):")
    print(def_rz.sort_values("opponent_factor", ascending=False).to_string(index=False))
