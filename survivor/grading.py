"""
grading.py -- Turns each survivor option into a 0-100 score and a letter grade.

  Score = base (this week's win chance)
        - future cost (using a team you'll need later)
        + injury points  (your key players out = minus, theirs out = plus)
        + situation points (rest, back-to-back road, time zones, early body-clock games)

All the point values live in survivor.py's settings block (GRADE_WEIGHTS).
Grades work like school: A = 90+, B = 80-89, C = 70-79, D = 60-69, F < 60.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from team_info import TEAM_INFO

B = "https://github.com/nflverse/nflverse-data/releases/download/"
STATS_URL = B + "stats_player/stats_player_week_{season}.parquet"
SNAPS_URL = B + "snap_counts/snap_counts_{season}.parquet"
INJ_URL = B + "injuries/injuries_{season}.parquet"
ROSTER_URL = B + "weekly_rosters/roster_weekly_{season}.parquet"

OL_POS = {"T", "G", "C", "OL", "OT", "OG"}
CB_POS = {"CB"}
RUSH_POS = {"DE", "DT", "DL", "NT", "OLB", "LB", "EDGE"}

DEFAULT_WEIGHTS = {
    # ---- injuries (points off for YOUR team; same points added if it's the OPPONENT's player)
    "qb": 15,              # starting QB
    "ol": 6,               # each starting offensive lineman
    "ol_two_plus": 6,      # extra hit when 2+ starting linemen are out
    "skill": 5,            # each of: top 2 WRs, top RB
    "pass_rusher": 4,      # each of the team's top 2 pass rushers
    "cb": 4,               # each starting cornerback (top 2)
    "questionable": 0.5,   # Questionable counts this fraction of the penalty (Out/Doubtful = full)
    "mirror_opponent": True,
    # ---- rest / travel (points for your team; the opposite for the other side)
    "rest_per_day": 3,     # per extra day of rest vs. the opponent
    "rest_cap": 12,
    "b2b_road": 8,         # opponent on back-to-back road games = +8 (you on one = -8)
    "time_zones": 5,       # road team crossing 2+ time zones
    "early_west": 5,       # West/Mountain road team in a 1 p.m. Eastern kickoff
    # ---- base score
    "base_offset": 10,     # base = win% + this  (so an 80% favorite starts at 90 = A-)
    "future_per_pct": 0.5, # points off per 1% of future survival odds lost
}

import re


def name_key(name) -> str:
    """'D.J. Reed Jr.' -> 'NAME:djreed' (snap counts don't always carry player IDs, so OL/CB match by name+team)."""
    n = re.sub(r"\b(jr|sr|ii|iii|iv|v)\b\.?", "", str(name).lower())
    return "NAME:" + re.sub(r"[^a-z]", "", n)


GRADES = [(97, "A+"), (93, "A"), (90, "A-"), (87, "B+"), (83, "B"), (80, "B-"),
          (77, "C+"), (73, "C"), (70, "C-"), (67, "D+"), (63, "D"), (60, "D-")]


def letter(score: float) -> str:
    for cut, g in GRADES:
        if score >= cut:
            return g
    return "F"


# =====================================================================
#  Key players + injury report
# =====================================================================
class SeasonData:
    """Everything needed to find key players and injuries for one season (downloaded once)."""

    def __init__(self, season: int):
        self.season = season
        st = pd.read_parquet(STATS_URL.format(season=season))
        self.stats = st[st.season_type == "REG"].copy()
        sn = pd.read_parquet(SNAPS_URL.format(season=season))
        sn = sn[sn.game_type == "REG"].copy()
        self.snaps = sn
        try:
            inj = pd.read_parquet(INJ_URL.format(season=season))
            if "game_type" in inj.columns:
                inj = inj[inj.game_type == "REG"]
            self.inj = inj[inj.report_status.isin(["Out", "Doubtful", "Questionable"])].copy()
        except Exception:
            self.inj = pd.DataFrame(columns=["team", "week", "gsis_id", "full_name", "position", "report_status"])

    def has_report(self, week: int) -> bool:
        return bool((self.inj.week == week).any())

    def key_players(self, week: int) -> dict:
        """{team: {gsis_id: (role, name)}} using only games BEFORE `week`."""
        st = self.stats[self.stats.week < week]
        sn = self.snaps[self.snaps.week < week]
        key: dict = {}

        def add(team, pid, role, name):
            if isinstance(pid, str) and pid:
                key.setdefault(team, {})[pid] = (role, name)

        for team, d in st.groupby("team"):
            # starting QB = most attempts in the team's latest game
            q = d[d.position == "QB"]
            if not q.empty:
                last = q[q.week == q.week.max()].sort_values("attempts", ascending=False).iloc[0]
                add(team, last.player_id, "QB", last.player_display_name)
            # top 2 WRs by targets
            wr = d[d.position == "WR"].groupby(["player_id", "player_display_name"]).targets.sum()
            for (pid, nm), v in wr.sort_values(ascending=False).head(2).items():
                if v >= 8:
                    add(team, pid, "WR", nm)
            # top RB by touches
            rb = d[d.position == "RB"].assign(t=lambda x: x.carries.fillna(0) + x.targets.fillna(0))
            rb = rb.groupby(["player_id", "player_display_name"]).t.sum()
            for (pid, nm), v in rb.sort_values(ascending=False).head(1).items():
                if v >= 15:
                    add(team, pid, "RB", nm)
            # top 2 pass rushers by sacks + half of QB hits
            pr = d[d.position.isin(RUSH_POS)].assign(p=lambda x: x.def_sacks.fillna(0) + 0.5 * x.def_qb_hits.fillna(0))
            pr = pr.groupby(["player_id", "player_display_name"]).p.sum()
            for (pid, nm), v in pr.sort_values(ascending=False).head(2).items():
                if v >= 1:
                    add(team, pid, "Pass rusher", nm)

        for team, d in sn.groupby("team"):
            # starting OL: played most offensive snaps when active
            ol = d[d.position.isin(OL_POS) & (d.offense_snaps > 0)]
            g = ol.groupby("player").agg(pct=("offense_pct", "mean"), n=("offense_snaps", "sum"))
            for nm, r in g[g.pct >= 0.6].sort_values("n", ascending=False).head(5).iterrows():
                add(team, name_key(nm), "OL", nm)
            # starting CBs
            cb = d[d.position.isin(CB_POS) & (d.defense_snaps > 0)]
            g = cb.groupby("player").agg(pct=("defense_pct", "mean"), n=("defense_snaps", "sum"))
            for nm, r in g[g.pct >= 0.6].sort_values("n", ascending=False).head(2).iterrows():
                add(team, name_key(nm), "CB", nm)
        return key

    def injury_hits(self, week: int, key: dict | None = None) -> dict:
        """{team: [(role, name, status)]} for key players on this week's report."""
        key = key if key is not None else self.key_players(week)
        rep = self.inj[self.inj.week == week]
        out: dict = {}
        for r in rep.itertuples():
            tk = key.get(r.team, {})
            k = tk.get(r.gsis_id) or tk.get(name_key(r.full_name))
            if k:
                out.setdefault(r.team, []).append((k[0], k[1], r.report_status))
        return out


def injury_points(hits: list, w: dict) -> list:
    """Penalty lines [(text, points)] for one team's injured key players (points are negative)."""
    lines = []
    ol_load = 0.0
    role_pts = {"QB": w["qb"], "OL": w["ol"], "WR": w["skill"], "RB": w["skill"],
                "Pass rusher": w["pass_rusher"], "CB": w["cb"]}
    for role, name, status in hits:
        f = w["questionable"] if status == "Questionable" else 1.0
        pts = -role_pts[role] * f
        lines.append((f"{role} {name} ({status})", pts))
        if role == "OL":
            ol_load += f
    if ol_load >= 2:
        lines.append(("2+ starting OL out", -w["ol_two_plus"]))
    return lines


# =====================================================================
#  Rest / travel
# =====================================================================
def prev_game_away(sched: pd.DataFrame, team: str, season: int, week: int):
    g = sched[(sched.season == season) & (sched.week < week) &
              ((sched.home_team == team) | (sched.away_team == team))]
    if g.empty:
        return None
    return g.sort_values("week").iloc[-1].away_team == team


def situation_points(row, team: str, sched: pd.DataFrame, w: dict, prev_away: dict | None = None) -> list:
    """[(text, points)] for `team` in schedule `row`. Zero-sum: the opponent gets the opposite."""
    home = row.home_team == team
    opp = row.away_team if home else row.home_team
    lines = []

    # rest difference
    r_t = row.home_rest if home else row.away_rest
    r_o = row.away_rest if home else row.home_rest
    if pd.notna(r_t) and pd.notna(r_o) and r_t != r_o:
        pts = float(np.clip((r_t - r_o) * w["rest_per_day"], -w["rest_cap"], w["rest_cap"]))
        lines.append((f"Rest {int(r_t)} days vs {int(r_o)}", pts))

    # back-to-back road games (only the road team can be on one)
    def b2b(t):
        if prev_away is not None:
            return prev_away.get((t, row.season, row.week), False)
        return bool(prev_game_away(sched, t, row.season, row.week))
    away_t = row.away_team
    if row.location == "Home" and b2b(away_t):
        lines.append((f"{away_t} on back-to-back road games", w["b2b_road"] if away_t == opp else -w["b2b_road"]))

    # travel: only for true home games
    if row.location == "Home" and row.home_team in TEAM_INFO and away_t in TEAM_INFO:
        tz = abs(TEAM_INFO[row.home_team][2] - TEAM_INFO[away_t][2])
        if tz >= 2:
            lines.append((f"{away_t} crossing {tz} time zones", w["time_zones"] if away_t == opp else -w["time_zones"]))
        if TEAM_INFO[away_t][2] <= -7 and TEAM_INFO[row.home_team][2] == -5 and str(row.gametime) < "14:00":
            lines.append((f"{away_t} (West) in a 1 p.m. ET game", w["early_west"] if away_t == opp else -w["early_west"]))
    return lines


def grade_team(p_win: float, future_pct: float, inj_own: list, inj_opp: list, situation: list, w: dict) -> dict:
    """Combine everything into a score + breakdown for one team."""
    parts = [(f"Win chance {p_win:.1%}", 100 * p_win + w["base_offset"])]
    if future_pct > 0.5:
        parts.append((f"Future cost (-{future_pct:.0f}% later odds)", -w["future_per_pct"] * future_pct))
    parts += injury_points(inj_own, w)
    if w.get("mirror_opponent", True):
        parts += [(f"Opp {t}", -p) for t, p in injury_points(inj_opp, w)]
    parts += situation
    score = sum(p for _, p in parts)
    inj_total = sum(p for t, p in parts if t.startswith(("QB", "OL", "WR", "RB", "Pass", "CB", "2+", "Opp")))
    sit_total = sum(p for _, p in situation)
    return dict(score=score, grade=letter(score), parts=parts, inj=inj_total, sit=sit_total)
