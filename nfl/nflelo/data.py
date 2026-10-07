"""Loading nflverse play-by-play + schedules, and reducing them to one row
per team-game with everything the rating engine needs."""

from __future__ import annotations

import os
import pandas as pd
import numpy as np

from . import config

TEAM_FIX = {"SD": "LAC", "STL": "LA", "SL": "LA", "OAK": "LV", "JAC": "JAX",
            "ARZ": "ARI", "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "LAR": "LA"}


def standardize(s):
    """nflfastR back-maps relocated franchises; nfldata does not. Align them."""
    return s.replace(TEAM_FIX)


CACHE_DIR = os.environ.get("NFLELO_CACHE", os.path.expanduser("~/.nflelo_cache"))


def _cache_path(name: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    return os.path.join(CACHE_DIR, name)


def load_pbp(seasons, refresh_last=True) -> pd.DataFrame:
    """Play-by-play for the given seasons. Completed seasons are cached on disk;
    the current season is re-pulled every run so weekly updates land."""
    frames = []
    cols = [
        "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam",
        "play_type", "qb_dropback", "rush_attempt", "pass_attempt", "sack",
        "qb_hit", "epa", "down", "ydstogo", "interception", "fumble",
        "fumble_lost", "passer_player_name", "passer_player_id",
        "rusher_player_name", "complete_pass", "yards_gained", "air_yards",
        "fourth_down_converted", "fourth_down_failed", "wp", "qtr",
        "third_down_converted", "third_down_failed", "half_seconds_remaining",
    ]
    for season in seasons:
        path = _cache_path(f"pbp_{season}.parquet")
        stale = refresh_last and season == max(seasons)
        if os.path.exists(path) and not stale:
            df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(config.PBP_URL.format(season=season))
            keep = [c for c in cols if c in df.columns]
            df = df[keep]
            df.to_parquet(path, index=False)
        frames.append(df)
    pbp = pd.concat(frames, ignore_index=True)
    return pbp[pbp["season_type"] == "REG"] if "season_type" in pbp else pbp


def load_games(refresh=True) -> pd.DataFrame:
    """Schedule + results + listed starting QBs + coaches."""
    path = _cache_path("games.csv")
    if refresh or not os.path.exists(path):
        g = pd.read_csv(config.GAMES_URL)
        g.to_csv(path, index=False)
    else:
        g = pd.read_csv(path)
    g = g[g["game_type"] == "REG"].copy()
    for col in ("home_team", "away_team"):
        g[col] = standardize(g[col])
    return g


FTN_URL = "https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
INJURY_URL = "https://github.com/nflverse/nflverse-data/releases/download/injuries/injuries_{season}.parquet"
SNAP_URL = "https://github.com/nflverse/nflverse-data/releases/download/snap_counts/snap_counts_{season}.parquet"
DEPTH_URL = "https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{season}.parquet"
FTN_MIN_SEASON = 2022  # nflverse doesn't publish FTN charting before this


def _cached_parquet(url_template, season, name, refresh_current=True, current_season=None):
    path = _cache_path(f"{name}_{season}.parquet")
    stale = refresh_current and season == current_season
    if os.path.exists(path) and not stale:
        return pd.read_parquet(path)
    try:
        df = pd.read_parquet(url_template.format(season=season))
    except Exception:
        return None
    df.to_parquet(path, index=False)
    return df


def load_ftn(seasons) -> pd.DataFrame | None:
    """FTN's manually charted play context: blitzers, pass rushers, QB-fault
    sacks, play action / motion / RPO / screen tags. Nothing here is a direct
    'pressure' flag -- FTN doesn't publish one -- so this refines sack
    attribution and scheme tendencies rather than replacing the pressure
    proxy in team_game_units."""
    frames = []
    cur = max(seasons)
    for s in seasons:
        if s < FTN_MIN_SEASON:
            continue
        df = _cached_parquet(FTN_URL, s, "ftn", current_season=cur)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_injuries(seasons) -> pd.DataFrame | None:
    frames = []
    cur = max(seasons)
    for s in seasons:
        df = _cached_parquet(INJURY_URL, s, "injuries", current_season=cur)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_snap_counts(seasons) -> pd.DataFrame | None:
    frames = []
    cur = max(seasons)
    for s in seasons:
        df = _cached_parquet(SNAP_URL, s, "snaps", current_season=cur)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def load_depth_charts(seasons) -> pd.DataFrame | None:
    frames = []
    cur = max(seasons)
    for s in seasons:
        df = _cached_parquet(DEPTH_URL, s, "depth", current_season=cur)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else None


def ftn_team_game(ftn: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game: blitz rate faced/generated, QB-fault sack share,
    play-action / motion / RPO rate. Joined onto pbp via nflverse play id."""
    key = pbp[["game_id", "play_id", "posteam", "defteam", "qb_dropback", "sack"]].copy()
    m = ftn.merge(key, left_on=["nflverse_game_id", "nflverse_play_id"],
                  right_on=["game_id", "play_id"], how="inner")
    m = m[m["posteam"].notna()]
    m["team"] = standardize(m["posteam"])
    m["opp"] = standardize(m["defteam"])
    m["is_dropback"] = m["qb_dropback"].fillna(0).astype(bool)

    off = m.groupby(["game_id", "team"]).apply(
        lambda g: pd.Series({
            "blitz_rate_faced": g.loc[g.is_dropback, "n_blitzers"].gt(0).mean()
                if g.is_dropback.any() else np.nan,
            "qb_fault_sack_rate": (g["is_qb_fault_sack"].sum() / max(g["sack"].sum(), 1))
                if g["sack"].sum() > 0 else np.nan,
            "play_action_rate": g.loc[g.is_dropback, "is_play_action"].mean()
                if g.is_dropback.any() else np.nan,
            "motion_rate": g["is_motion"].mean(),
            "rpo_rate": g.loc[g.is_dropback, "is_rpo"].mean()
                if g.is_dropback.any() else np.nan,
            "drop_rate": g.loc[g.is_dropback, "is_drop"].mean()
                if g.is_dropback.any() else np.nan,
        }), include_groups=False).reset_index()

    dfn = m.groupby(["game_id", "opp"]).apply(
        lambda g: pd.Series({
            "blitz_rate_generated": g.loc[g.is_dropback, "n_blitzers"].gt(0).mean()
                if g.is_dropback.any() else np.nan,
            "avg_pass_rushers": g.loc[g.is_dropback, "n_pass_rushers"].mean(),
        }), include_groups=False).reset_index().rename(columns={"opp": "team"})

    return off.merge(dfn, on=["game_id", "team"], how="outer")


def team_game_units(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, offense team) with the four unit signals and the
    supporting metrics. EPA is from the offense's point of view."""
    p = pbp[pbp["posteam"].notna() & pbp["epa"].notna()].copy()

    # A dropback is pass attempt + sack + scramble. nflfastR flags it directly.
    p["is_pass"] = p["qb_dropback"].fillna(0).astype(bool)
    p["is_rush"] = (p["rush_attempt"].fillna(0).astype(bool)) & (~p["is_pass"])
    p["early_down"] = p["down"].isin([1, 2])
    p["late_down"] = p["down"].isin([3, 4])
    p["qb_hit"] = p["qb_hit"].fillna(0)
    p["sack"] = p["sack"].fillna(0)
    p["fumble"] = p["fumble"].fillna(0)
    p["interception"] = p["interception"].fillna(0)
    p["fumble_lost"] = p["fumble_lost"].fillna(0)

    def agg(g: pd.DataFrame) -> pd.Series:
        pas, rsh = g[g.is_pass], g[g.is_rush]
        n_pass, n_rush = len(pas), len(rsh)
        early, late = g[g.early_down], g[g.late_down]
        return pd.Series({
            "season": g["season"].iloc[0],
            "week": g["week"].iloc[0],
            "defteam": g["defteam"].iloc[0],
            "n_pass": n_pass,
            "n_rush": n_rush,
            "pass_epa": pas["epa"].mean() if n_pass else np.nan,
            "rush_epa": rsh["epa"].mean() if n_rush else np.nan,
            # --- pressure proxy: free-data stand-in for charted pressures.
            # Sacks + QB hits per dropback. Correlates ~.75 with PFF pressure.
            "pressures_allowed": (pas["sack"].sum() + pas["qb_hit"].sum()),
            "sacks_allowed": pas["sack"].sum(),
            # --- down leverage: late-down efficiency above own early-down base
            "early_epa": early["epa"].mean() if len(early) else np.nan,
            "late_epa": late["epa"].mean() if len(late) else np.nan,
            "third_att": int(g["third_down_converted"].fillna(0).sum()
                             + g["third_down_failed"].fillna(0).sum()),
            "third_conv": int(g["third_down_converted"].fillna(0).sum()),
            # --- turnovers: track fumbles FORCED separately from RECOVERED,
            # because recovery is a coin flip and EPA already paid for it.
            "ints": g["interception"].sum(),
            "fumbles": g["fumble"].sum(),
            "fumbles_lost": g["fumble_lost"].sum(),
            # --- coaching behavior
            "fourth_go": int(g["fourth_down_converted"].fillna(0).sum()
                             + g["fourth_down_failed"].fillna(0).sum()),
            "plays": len(g),
            "passer": (pas.groupby("passer_player_id")["epa"].count().idxmax()
                       if pas["passer_player_id"].notna().any() else None),
            "passer_name": (pas.groupby("passer_player_name")["epa"].count().idxmax()
                            if pas["passer_player_name"].notna().any() else None),
            "qb_epa": pas["epa"].mean() if n_pass else np.nan,
        })

    out = p.groupby(["game_id", "posteam"]).apply(agg, include_groups=False).reset_index()
    out = out.rename(columns={"posteam": "team"})
    out["team"] = standardize(out["team"])
    out["defteam"] = standardize(out["defteam"])
    out["pressure_rate_allowed"] = out["pressures_allowed"] / out["n_pass"].clip(lower=1)
    out["sack_rate_allowed"] = out["sacks_allowed"] / out["n_pass"].clip(lower=1)
    out["down_leverage"] = out["late_epa"] - out["early_epa"]
    out["pass_rate"] = out["n_pass"] / (out["n_pass"] + out["n_rush"]).clip(lower=1)
    return out


def attach_ftn(units: pd.DataFrame, ftn_team_game: pd.DataFrame | None) -> pd.DataFrame:
    """Left-join FTN charting columns onto the unit table. Rows before 2022,
    or any season FTN charting isn't published for, simply get NaN in the new
    columns -- observe_game() already treats missing FTN fields as 'skip'."""
    if ftn_team_game is None:
        return units
    return units.merge(ftn_team_game, on=["game_id", "team"], how="left")


def qb_dropback_table(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per-QB dropback EPA, for the starter-change adjustment."""
    p = pbp[pbp["qb_dropback"].fillna(0).astype(bool) & pbp["epa"].notna()].copy()
    p = p[p["passer_player_name"].notna()]
    q = (p.groupby(["season", "week", "posteam", "passer_player_name"])
           .agg(db=("epa", "size"), epa=("epa", "mean"),
                sacks=("sack", "sum"))
           .reset_index()
           .rename(columns={"passer_player_name": "qb", "posteam": "team"}))
    return q
