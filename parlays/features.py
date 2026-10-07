"""
features.py -- Building blocks shared by the TD and yards models.

  recency_weights   recent games count more than older ones
  snap_shares       how much of the offense each player is on the field for
  opponent_factors  how a defense does against each position (RB/WR/TE/QB)
  redistribute      when a player is Out, hand his opportunities to the
                    teammates at his position who'd be on the field instead
"""

from __future__ import annotations
import numpy as np
import pandas as pd

BASE = "https://github.com/nflverse/nflverse-data/releases/download/"
SNAPS_URL = BASE + "snap_counts/snap_counts_{season}.parquet"
ROSTER_URL = BASE + "weekly_rosters/roster_weekly_{season}.parquet"
INJURIES_URL = BASE + "injuries/injuries_{season}.parquet"

POS_GROUPS = ("QB", "RB", "WR", "TE")


def recency_weights(weeks: pd.Series, target_week: int, decay: float) -> np.ndarray:
    """Weight of each game: 1.0 for last week, decay for the week before, decay^2 ..."""
    return decay ** (target_week - 1 - weeks.to_numpy())


# ------------------------------------------------------------- snaps
def load_snaps(season: int) -> pd.DataFrame:
    """Offensive snap % per player-game, keyed by gsis player_id."""
    s = pd.read_parquet(SNAPS_URL.format(season=season),
                        columns=["season", "game_type", "week", "pfr_player_id", "team", "offense_pct"])
    s = s[s["game_type"] == "REG"]
    r = pd.read_parquet(ROSTER_URL.format(season=season), columns=["gsis_id", "pfr_id"]).dropna().drop_duplicates()
    s = s.merge(r, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    return s.rename(columns={"gsis_id": "player_id"})[["player_id", "week", "team", "offense_pct"]]


def snap_shares(snaps: pd.DataFrame, target_week: int, decay: float = 0.7) -> dict:
    """Recency-weighted offensive snap share per player, from games before target_week."""
    s = snaps[snaps["week"] < target_week]
    if s.empty:
        return {}
    s = s.assign(w=recency_weights(s["week"], target_week, decay))
    g = s.groupby("player_id").apply(lambda d: np.average(d["offense_pct"], weights=d["w"]), include_groups=False)
    return g.to_dict()


# ------------------------------------------------------------- opponents
def opponent_factors(stats: pd.DataFrame, target_week: int, value_col: str,
                     shrink_games: float = 4.0) -> dict:
    """
    {(defense_team, position): factor}. 1.0 = average, 1.2 = this defense
    gives up 20% more of value_col to that position than an average one.
    Uses this season's games before target_week, pulled toward 1.0 when
    there are only a few games so far.
    """
    s = stats[(stats["week"] < target_week) & stats["position"].isin(POS_GROUPS)]
    if s.empty:
        return {}
    per_game = s.groupby(["opponent_team", "position", "week"])[value_col].sum().reset_index()
    allowed = per_game.groupby(["opponent_team", "position"]).agg(v=(value_col, "mean"), n=("week", "nunique")).reset_index()
    league = allowed.groupby("position")["v"].mean()
    out = {}
    for r in allowed.itertuples():
        avg = league.get(r.position, 0)
        if avg <= 0:
            continue
        raw = r.v / avg
        out[(r.opponent_team, r.position)] = (r.n * raw + shrink_games) / (r.n + shrink_games)
    return out


# ------------------------------------------------------------- injuries
def out_players(season: int, week: int, injuries: pd.DataFrame | None = None) -> set:
    """gsis ids listed Out or Doubtful for that week."""
    if injuries is None:
        try:
            injuries = pd.read_parquet(INJURIES_URL.format(season=season))
        except Exception:
            return set()
    i = injuries[(injuries["season"] == season) & (injuries["week"] == week)]
    return set(i.loc[i["report_status"].isin(["Out", "Doubtful"]), "gsis_id"].dropna())


def redistribute(values: dict, team_of: dict, pos_of: dict, out: set, snaps: dict,
                 fraction: float) -> dict:
    """
    values: {player_id: opportunity number} (e.g. targets per game, or TD share).
    For each player who is Out, give `fraction` of his number to active
    teammates at the same position, split by their snap share (the guys
    who'd actually take his snaps). Returns a new dict for active players.
    """
    new = {p: v for p, v in values.items() if p not in out}
    if fraction <= 0:
        return new
    for o in out:
        if o not in values or o not in team_of:
            continue
        team, pos = team_of[o], pos_of.get(o)
        mates = [p for p in new if team_of.get(p) == team and pos_of.get(p) == pos]
        if not mates:
            continue
        w = np.array([max(snaps.get(p, 0.0), 0.02) for p in mates])
        w = w / w.sum()
        for p, share in zip(mates, w):
            new[p] += fraction * values[o] * share
    return new
