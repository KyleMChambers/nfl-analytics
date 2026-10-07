"""Optional: real charted pressure numbers from Pro Football Reference.

The model ships with a free proxy for pressure (sacks + QB hits per dropback,
straight out of play-by-play), which correlates well with charted pressure but
misses hurries entirely. PFR publishes actual pressure counts on its advanced
defense page. This module pulls them when you want the sharper input.

Be a good citizen: PFR rate-limits hard. Results are cached; call it once a
week, not once a game. If the fetch fails for any reason the model silently
falls back to the play-by-play proxy, so nothing here is load-bearing.
"""

from __future__ import annotations

import os
import time
import pandas as pd

from . import config as C
from .data import _cache_path, standardize

PFR_TEAM = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS", "Oakland Raiders": "LV",
    "San Diego Chargers": "LAC", "St. Louis Rams": "LA",
}


def team_pressures(season: int, max_age_hours: float = 24.0):
    """{team: pressures per opponent dropback} or None if unavailable."""
    path = _cache_path(f"pfr_pressure_{season}.csv")
    if os.path.exists(path) and (time.time() - os.path.getmtime(path)) < max_age_hours * 3600:
        df = pd.read_csv(path)
    else:
        try:
            tables = pd.read_html(C.PFR_DEF_URL.format(season=season))
        except Exception:
            return None
        df = None
        for t in tables:
            cols = [str(c[-1]) if isinstance(c, tuple) else str(c) for c in t.columns]
            if "Prss" in cols or "Prss%" in cols:
                t.columns = cols
                df = t
                break
        if df is None:
            return None
        df = df[df["Tm"].isin(PFR_TEAM)].copy()
        df["team"] = standardize(df["Tm"].map(PFR_TEAM))
        keep = ["team"] + [c for c in ("Prss", "Prss%", "Att", "Sk") if c in df.columns]
        df = df[keep]
        df.to_csv(path, index=False)

    if "Prss%" in df.columns:
        rate = pd.to_numeric(df["Prss%"], errors="coerce") / 100.0
    elif "Prss" in df.columns and "Att" in df.columns:
        rate = pd.to_numeric(df["Prss"], errors="coerce") / pd.to_numeric(df["Att"], errors="coerce")
    else:
        return None
    return dict(zip(df["team"], rate))


def blend_into(model, season, weight=0.6):
    """Overwrite the proxy pressure-generation rate with the charted one.

    weight=0.6 keeps 40% of the play-by-play proxy, which is game-recency
    weighted; the PFR number is a flat season total with no recency at all.
    """
    real = team_pressures(season)
    if not real:
        return False
    for team, r in real.items():
        st = model.teams.get(team)
        if st is None or r != r:
            continue
        cur = st.press_def
        st.press_def = r if cur != cur else weight * r + (1 - weight) * cur
    return True
