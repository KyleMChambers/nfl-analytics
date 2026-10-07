"""
context.py -- Everything you asked to see for each matchup.

These are shown next to each pick as context. The ones that feed the grade
(injuries, rest, back-to-back road, time zones, early West Coast games) are
scored in grading.py; run backtest_grades.py to see how they've held up.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import requests

from team_info import TEAM_INFO, miles_between

B = "https://github.com/nflverse/nflverse-data/releases/download/"
PBP_URL = B + "pbp/play_by_play_{season}.parquet"
STATS_URL = B + "stats_player/stats_player_week_{season}.parquet"
INJ_URL = B + "injuries/injuries_{season}.parquet"
SNAPS_URL = B + "snap_counts/snap_counts_{season}.parquet"
ROSTER_URL = B + "weekly_rosters/roster_weekly_{season}.parquet"


class Context:
    def __init__(self, sched: pd.DataFrame, season: int, week: int):
        self.season, self.week = season, week
        self.sched = sched
        self.played = sched[(sched["season"] == season) & (sched["week"] < week) & sched["result"].notna()]
        self._team_lines()
        self._defense_yards()
        self._qbs()
        self._coaches()
        self._injuries()

    # ---------------- team results
    def _team_lines(self):
        rows = []
        for r in self.played.itertuples():
            rows.append((r.home_team, r.home_score, r.away_score))
            rows.append((r.away_team, r.away_score, r.home_score))
        d = pd.DataFrame(rows, columns=["team", "pf", "pa"])
        d["w"], d["l"], d["t"] = (d.pf > d.pa).astype(int), (d.pf < d.pa).astype(int), (d.pf == d.pa).astype(int)
        self.team = d.groupby("team").agg(w=("w", "sum"), l=("l", "sum"), t=("t", "sum"),
                                          ppg=("pf", "mean"), papg=("pa", "mean"))

    def record(self, t):
        if t not in self.team.index:
            return "0-0"
        r = self.team.loc[t]
        return f"{int(r.w)}-{int(r.l)}" + (f"-{int(r.t)}" if r.t else "")

    def ppg(self, t):
        return self.team.at[t, "ppg"] if t in self.team.index else np.nan

    def papg(self, t):
        return self.team.at[t, "papg"] if t in self.team.index else np.nan

    # ---------------- defense yards allowed
    def _defense_yards(self):
        try:
            p = pd.read_parquet(PBP_URL.format(season=self.season),
                                columns=["season_type", "week", "game_id", "defteam", "yards_gained", "play_type"])
            p = p[(p.season_type == "REG") & (p.week < self.week) & p.play_type.isin(["pass", "run"])]
            g = p.groupby("defteam").agg(y=("yards_gained", "sum"), n=("game_id", "nunique"))
            self.def_ypg = (g.y / g.n).to_dict()
        except Exception:
            self.def_ypg = {}

    # ---------------- quarterbacks
    def _qbs(self):
        self.qb = {}
        try:
            s = pd.read_parquet(STATS_URL.format(season=self.season))
            s = s[(s.season_type == "REG") & (s.week < self.week) & (s.position == "QB")]
        except Exception:
            return
        for team, d in s.groupby("team"):
            last = d[d.week == d.week.max()].sort_values("attempts", ascending=False).iloc[0]
            q = d[d.player_id == last.player_id]
            att = q.attempts.sum()
            if att < 10:
                continue
            a = np.clip((q.completions.sum() / att - 0.3) * 5, 0, 2.375)
            b = np.clip((q.passing_yards.sum() / att - 3) * 0.25, 0, 2.375)
            c = np.clip(q.passing_tds.sum() / att * 20, 0, 2.375)
            e = np.clip(2.375 - q.passing_interceptions.sum() / att * 25, 0, 2.375)
            self.qb[team] = (last.player_display_name, (a + b + c + e) / 6 * 100, last.player_id)

    # ---------------- coaches
    def _coaches(self):
        g = self.sched[self.sched["result"].notna() & ((self.sched.season < self.season) |
                       ((self.sched.season == self.season) & (self.sched.week < self.week)))]
        rows = [(c, int(r > 0), int(r < 0)) for c, r in zip(g.home_coach, g.result)] + \
               [(c, int(r < 0), int(r > 0)) for c, r in zip(g.away_coach, g.result)]
        d = pd.DataFrame(rows, columns=["coach", "w", "l"]).groupby("coach").sum()
        self.coach = d.to_dict("index")

    def coach_record(self, name):
        if not name or name not in self.coach:
            return f"{name or '?'} (new)"
        r = self.coach[name]
        pct = r["w"] / max(r["w"] + r["l"], 1)
        return f"{name} {r['w']}-{r['l']} ({pct:.0%})"

    # ---------------- injuries (key players only)
    def _injuries(self):
        self.inj = {}
        try:
            inj = pd.read_parquet(INJ_URL.format(season=self.season))
            inj = inj[(inj.week == self.week) & inj.report_status.isin(["Out", "Doubtful", "Questionable"])]
            snaps = pd.read_parquet(SNAPS_URL.format(season=self.season))
            snaps = snaps[(snaps.game_type == "REG") & (snaps.week < self.week)]
            rost = pd.read_parquet(ROSTER_URL.format(season=self.season), columns=["gsis_id", "pfr_id"]).dropna().drop_duplicates()
            share = snaps.merge(rost, left_on="pfr_player_id", right_on="pfr_id")
            share = share.assign(pct=share[["offense_pct", "defense_pct"]].max(axis=1)).groupby("gsis_id")["pct"].mean()
        except Exception:
            return
        qb_ids = {v[2] for v in self.qb.values()}
        for r in inj.itertuples():
            starter = share.get(r.gsis_id, 0) >= 0.6 or r.gsis_id in qb_ids
            if not starter:
                continue
            tag = f"{r.full_name} ({r.position}, {r.report_status})"
            if r.gsis_id in qb_ids:
                tag = "STARTING QB " + tag
            self.inj.setdefault(r.team, []).append(tag)

    # ---------------- travel / schedule spot
    def prev_was_road(self, team):
        g = self.sched[(self.sched.season == self.season) & (self.sched.week < self.week) &
                       ((self.sched.home_team == team) | (self.sched.away_team == team))]
        if g.empty:
            return False
        last = g.sort_values("week").iloc[-1]
        return last.away_team == team

    def travel(self, away, home):
        if away not in TEAM_INFO or home not in TEAM_INFO:
            return None
        return miles_between(away, home), abs(TEAM_INFO[home][2] - TEAM_INFO[away][2])

    def climate_flag(self, away, home, gameday, temp_low=None):
        if away not in TEAM_INFO or home not in TEAM_INFO:
            return None
        _, _, _, a_clim, a_in = TEAM_INFO[away]
        _, _, _, h_clim, h_in = TEAM_INFO[home]
        month = pd.to_datetime(gameday).month
        cold = (not h_in) and h_clim == "cold" and ((temp_low is not None and temp_low < 40) or
                                                     (temp_low is None and month in (12, 1)))
        if cold and (a_clim == "warm" or a_in):
            return f"{away} (warm/dome team) playing in the cold"
        hot = (not h_in) and h_clim == "warm" and month in (9, 10)
        if hot and a_clim == "cold" and not a_in:
            return f"{away} (cold-weather team) playing in the heat"
        return None

    # ---------------- weather (Open-Meteo, free, no key; ~2 weeks ahead)
    def weather(self, home, gameday):
        if home not in TEAM_INFO:
            return None, "unknown"
        lat, lon, _, _, indoor = TEAM_INFO[home]
        if indoor:
            return None, "indoors"
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", timeout=10, params={
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "start_date": gameday, "end_date": gameday})
            d = r.json()["daily"]
            hi, lo = d["temperature_2m_max"][0], d["temperature_2m_min"][0]
            rain, wind = d["precipitation_probability_max"][0], d["wind_speed_10m_max"][0]
            return lo, f"{lo:.0f}-{hi:.0f}F, {rain}% rain, wind {wind:.0f} mph"
        except Exception:
            return None, "forecast not available yet"
