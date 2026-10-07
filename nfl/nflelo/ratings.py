"""The rating engine.

Four ratings per team, all in EPA-per-play (= points per play) space:

    pass_off, rush_off   positive = generates more than league-average EPA
    pass_def, rush_def   positive = ALLOWS more than average (i.e. bad defense)

Each game, for each phase, we predict the offense's EPA/play as

    pred = league_mean(phase) + offense_rating + defense_rating

and push the residual into both ratings, Elo-style, split by K. That keeps the
familiar Elo mechanic (expectation -> surprise -> weighted update) while giving
you the four separate unit numbers you asked for, which a single-number Elo
updated on final score cannot produce.

Layered on top are the residual signals that raw EPA either misses or lies
about: pressure generation, late-down leverage regression, turnover luck,
coaching behavior, and quarterback identity.
"""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import pandas as pd

from . import config as C


def _shrink(value, n, prior, k):
    """Pull a small-sample stat toward a prior: n/(n+k) of the way to observed."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return prior
    w = n / (n + k)
    return w * value + (1 - w) * prior


class TeamState:
    __slots__ = ("pass_off", "rush_off", "pass_def", "rush_def", "games",
                 "press_def", "press_off", "lev_off", "lev_def",
                 "go_rate", "pace", "pass_rate", "qb_recent",
                 "qb_fault_sack", "blitz_faced", "blitz_generated")

    def __init__(self):
        self.pass_off = self.rush_off = self.pass_def = self.rush_def = 0.0
        self.games = 0
        self.press_def = self.press_off = np.nan
        self.lev_off = self.lev_def = np.nan
        self.go_rate = self.pace = self.pass_rate = np.nan
        self.qb_recent = None
        # FTN-derived: of THIS team's own sacks allowed, what fraction were
        # the QB's fault (held the ball) vs the O-line's. Travels with the
        # QB rating on a starter change, same as qb_recent.
        self.qb_fault_sack = np.nan
        self.blitz_faced = np.nan      # offense: how often opponents blitz them
        self.blitz_generated = np.nan  # defense: how often they blitz opponents

    def ewma(self, attr, value, span=6.0):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        cur = getattr(self, attr)
        a = 2.0 / (span + 1.0)
        setattr(self, attr, value if (cur is None or np.isnan(cur))
                else (1 - a) * cur + a * value)

    def new_season(self):
        self.pass_off *= C.CARRYOVER_PASS
        self.pass_def *= C.CARRYOVER_PASS
        self.rush_off *= C.CARRYOVER_RUSH
        self.rush_def *= C.CARRYOVER_RUSH
        self.games = 0


class LeagueMeans:
    """Rolling league baselines so the model self-calibrates as the NFL drifts."""

    def __init__(self):
        self.pass_epa = 0.08
        self.rush_epa = -0.07
        self.press = 0.155
        self.lev = 0.0
        self.go = 2.0
        self.pace = 63.0
        self.pass_rate = 0.585

    def update(self, row, a=0.002):
        for attr, val in (("pass_epa", row.pass_epa), ("rush_epa", row.rush_epa),
                          ("press", row.pressure_rate_allowed),
                          ("lev", row.down_leverage), ("go", row.fourth_go),
                          ("pace", row.plays), ("pass_rate", row.pass_rate)):
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                setattr(self, attr, (1 - a) * getattr(self, attr) + a * val)


class QBBook:
    """Per-QB dropback EPA, shrunk to league mean and decayed over time.

    This is the single biggest lever in the whole model. A starter-to-backup
    swap moves a spread 3-7 points -- more than every box-score metric on the
    list combined -- so the rating has to travel with the player, not the team.
    """

    def __init__(self):
        self.epa = defaultdict(float)   # shrunk EPA/dropback above average
        self.db = defaultdict(float)    # effective dropback count
        self.qbfs = defaultdict(float)  # shrunk qb-fault-sack rate
        self.team_of = {}

    def rating(self, qb):
        if qb is None:
            return None
        return _shrink(self.epa.get(qb, 0.0), self.db.get(qb, 0.0), 0.0,
                       C.STABILIZE["qb"])

    def fault_sack_rate(self, qb):
        """Shrunk toward the league-average qb-fault share, not zero -- this
        is a rate, not a points-above-average number."""
        if qb is None or qb not in self.qbfs:
            return None
        return self.qbfs[qb]

    def update(self, qb, team, n_db, epa_above_avg, decay=0.985,
               qb_fault_sack_rate=None, n_sacks=0):
        if qb is None or n_db <= 0:
            return
        prev_db, prev_epa = self.db[qb] * decay, self.epa[qb]
        tot = prev_db + n_db
        self.epa[qb] = (prev_epa * prev_db + epa_above_avg * n_db) / max(tot, 1e-9)
        self.db[qb] = tot
        self.team_of[qb] = team
        if qb_fault_sack_rate is not None and not np.isnan(qb_fault_sack_rate) and n_sacks > 0:
            prev = self.qbfs.get(qb, 0.35)  # rough league-average prior
            w = min(n_sacks / (n_sacks + 6.0), 0.9)
            self.qbfs[qb] = (1 - w) * prev + w * qb_fault_sack_rate


class Model:
    def __init__(self, cfg=C):
        self.C = cfg
        self.teams = defaultdict(TeamState)
        self.lg = LeagueMeans()
        self.qbs = QBBook()
        self._season = None

    # ------------------------------------------------------------------ core
    def _k_mult(self, state):
        m = self.C.EARLY_K_MULT
        return 1.0 + (m - 1.0) * max(0.0, 1 - state.games / self.C.EARLY_K_DECAY_GAMES)

    def _turnover_luck_points(self, row):
        """Fumble RECOVERY is close to a coin flip, but the EPA of a lost fumble
        is fully charged to the offense. Strip the luck back out before the
        residual touches the rating, or you spend the season chasing variance."""
        expected_lost = 0.5 * float(row.fumbles)
        excess = float(row.fumbles_lost) - expected_lost
        return excess * self.C.POINTS_PER_TURNOVER * self.C.W_TURNOVER_LUCK

    def observe_game(self, off_row, def_row):
        """Update both teams from one completed game (two team-game rows)."""
        for row, opp in ((off_row, def_row), (def_row, off_row)):
            o, d = self.teams[row.team], self.teams[row.defteam]
            km = self._k_mult(o)

            luck_pts = self._turnover_luck_points(row)
            n_tot = max(row.n_pass + row.n_rush, 1)
            luck_per_play = luck_pts / n_tot  # add back: we were unlucky if >0

            if row.n_pass > 0 and not np.isnan(row.pass_epa):
                actual = row.pass_epa + luck_per_play
                pred = self.lg.pass_epa + o.pass_off + d.pass_def
                err = actual - pred
                w = min(row.n_pass / self.C.DROPBACKS_PER_GAME, 1.5)
                o.pass_off += self.C.K_PASS_OFF * km * err * w
                d.pass_def += self.C.K_PASS_DEF * km * err * w

            if row.n_rush > 0 and not np.isnan(row.rush_epa):
                actual = row.rush_epa + luck_per_play
                pred = self.lg.rush_epa + o.rush_off + d.rush_def
                err = actual - pred
                w = min(row.n_rush / self.C.RUSHES_PER_GAME, 1.5)
                o.rush_off += self.C.K_RUSH_OFF * km * err * w
                d.rush_def += self.C.K_RUSH_DEF * km * err * w

            # ---- auxiliary signals
            o.ewma("press_off", row.pressure_rate_allowed)
            d.ewma("press_def", row.pressure_rate_allowed)
            o.ewma("lev_off", row.down_leverage)
            d.ewma("lev_def", row.down_leverage)
            o.ewma("go_rate", row.fourth_go, span=10)
            o.ewma("pace", row.plays, span=10)
            o.ewma("pass_rate", row.pass_rate, span=8)
            o.games += 1

            # ---- FTN charting (optional columns; absent if not merged in)
            qbfs_rate = getattr(row, "qb_fault_sack_rate", None)
            blitz_faced = getattr(row, "blitz_rate_faced", None)
            blitz_gen = getattr(row, "blitz_rate_generated", None)
            o.ewma("qb_fault_sack", qbfs_rate, span=8)
            o.ewma("blitz_faced", blitz_faced, span=8)
            d.ewma("blitz_generated", blitz_gen, span=8)

            # ---- quarterback
            if row.passer and row.n_pass > 0 and not np.isnan(row.qb_epa):
                self.qbs.update(row.passer, row.team, row.n_pass,
                                row.qb_epa - self.lg.pass_epa,
                                qb_fault_sack_rate=qbfs_rate,
                                n_sacks=row.sacks_allowed)
                o.qb_recent = row.passer

            self.lg.update(row)

    # ------------------------------------------------------------- prediction
    def _play_mix(self, team):
        s = self.teams[team]
        pr = s.pass_rate if not np.isnan(s.pass_rate) else self.lg.pass_rate
        pr = self.C.PLAY_MIX_WEIGHT * pr + (1 - self.C.PLAY_MIX_WEIGHT) * self.lg.pass_rate
        total = self.C.DROPBACKS_PER_GAME + self.C.RUSHES_PER_GAME
        return pr * total, (1 - pr) * total

    def team_points(self, team):
        """Expected scoring margin this team produces against a league-average
        opponent on a neutral field. This is the power rating."""
        s = self.teams[team]
        n_p, n_r = self._play_mix(team)
        off = n_p * s.pass_off + n_r * s.rush_off
        dfn = -(n_p * s.pass_def + n_r * s.rush_def)
        return off + dfn

    def _pressure_edge(self, a, b):
        sa, sb = self.teams[a], self.teams[b]
        def g(x, fallback):
            return self.lg.press if (x is None or np.isnan(x)) else x
        a_def = g(sa.press_def, self.lg.press) - self.lg.press
        b_def = g(sb.press_def, self.lg.press) - self.lg.press
        a_off = g(sa.press_off, self.lg.press) - self.lg.press
        b_off = g(sb.press_off, self.lg.press) - self.lg.press
        # A benefits from pressuring B and from protecting its own QB.
        edge = (a_def - b_def) + (b_off - a_off)
        return 100.0 * edge  # percentage points

    def _leverage_regression(self, a, b):
        """Teams whose production is concentrated on third down give it back.
        Subtract the unsustainable part of the edge already baked into EPA."""
        sa, sb = self.teams[a], self.teams[b]
        def g(x):
            return self.lg.lev if (x is None or np.isnan(x)) else x
        a_edge = (g(sa.lev_off) - self.lg.lev) - (g(sb.lev_def) - self.lg.lev)
        b_edge = (g(sb.lev_off) - self.lg.lev) - (g(sa.lev_def) - self.lg.lev)
        return -(a_edge - b_edge)

    def _coach_edge(self, a, b):
        sa, sb = self.teams[a], self.teams[b]
        def z(x, mean, sd):
            return 0.0 if (x is None or np.isnan(x)) else (x - mean) / sd
        go = z(sa.go_rate, self.lg.go, 1.0) - z(sb.go_rate, self.lg.go, 1.0)
        pace = z(sa.pace, self.lg.pace, 5.0) - z(sb.pace, self.lg.pace, 5.0)
        return 0.75 * go + 0.25 * pace

    def _qb_fault_sack_edge(self, home, away):
        """A QB who takes the blame for more of his own sacks predicts worse
        future sack rate independent of his O-line. Compare each offense's
        QB-fault share (bad for them) against the opponent's own baseline,
        so a mobile, clean-pocket QB isn't punished for a leaky O-line and
        vice versa."""
        def g(state, qb_rating_fn, recent_qb):
            per_qb = qb_rating_fn(recent_qb)
            return per_qb if per_qb is not None else state.qb_fault_sack
        h, a = self.teams[home], self.teams[away]
        h_rate = g(h, self.qbs.fault_sack_rate, h.qb_recent)
        a_rate = g(a, self.qbs.fault_sack_rate, a.qb_recent)
        if h_rate is None or a_rate is None or np.isnan(h_rate) or np.isnan(a_rate):
            return 0.0
        # Higher qb-fault-sack rate is bad for that offense -> subtract.
        return -(h_rate - a_rate) * 100.0  # in percentage points

    def _blitz_mismatch(self, home, away):
        h, a = self.teams[home], self.teams[away]
        def g(x, fallback=0.30):
            return fallback if (x is None or np.isnan(x)) else x
        # Home offense benefits when away blitzes less than home is used to
        # handling well, and vice versa -- approximated as a simple mismatch
        # between how much pressure each side's scheme creates/faces.
        edge = (g(h.blitz_faced) - g(a.blitz_generated)) - (g(a.blitz_faced) - g(h.blitz_generated))
        return -edge * 100.0

    def _environment_shrink(self, roof):
        if roof in ("dome", "closed"):
            return self.C.DOME_MARGIN_SHRINK
        return 0.0

    def injury_points(self, team, injury_rows):
        """injury_rows: iterable of (position, status, snap_share) for this
        team's questionable/doubtful/out players this week. Returns points
        SUBTRACTED from that team's own projected output (always <= 0)."""
        total = 0.0
        for pos, status, snap_share in injury_rows:
            base = self.C.INJURY_POSITION_VALUE.get(pos)
            sw = self.C.INJURY_STATUS_WEIGHT.get(status)
            if base is None or sw is None or snap_share is None or np.isnan(snap_share):
                continue
            total -= base * sw * min(max(snap_share, 0.0), 1.0)
        return total

    def _qb_adjustment(self, team, starter):
        """Points added/removed because the listed starter differs from whoever
        the team's rating was actually built on."""
        if not starter:
            return 0.0
        s = self.teams[team]
        base = self.qbs.rating(s.qb_recent)
        new = self.qbs.rating(starter)
        if new is None:
            new = -0.06  # unrated QB: below-average prior, not average
        if base is None:
            return 0.0
        n_p, _ = self._play_mix(team)
        return self.C.W_QB * (new - base) * n_p

    def predict(self, home, away, neutral=False, home_rest=7, away_rest=7,
                div_game=False, home_qb=None, away_qb=None, roof=None,
                home_injuries=(), away_injuries=()):
        """Returns the projected home margin and a breakdown of where it came from."""
        base = self.team_points(home) - self.team_points(away)
        hfa = 0.0 if neutral else self.C.HOME_FIELD_POINTS
        rest = self.C.REST_POINTS_PER_DAY * float(
            np.clip(home_rest - away_rest, -self.C.REST_DIFF_CAP, self.C.REST_DIFF_CAP))
        press = self.C.W_PRESSURE * self._pressure_edge(home, away)
        lev = self.C.W_THIRD_DOWN * self._leverage_regression(home, away)
        coach = self.C.W_COACH * self._coach_edge(home, away)
        qb = self._qb_adjustment(home, home_qb) - self._qb_adjustment(away, away_qb)
        qb_fault = self.C.W_QB_FAULT_SACK * self._qb_fault_sack_edge(home, away)
        blitz = self.C.W_BLITZ * self._blitz_mismatch(home, away)
        injuries = (self.injury_points(home, home_injuries)
                    - self.injury_points(away, away_injuries))

        margin = base + hfa + rest + press + lev + coach + qb + qb_fault + blitz + injuries
        if div_game:
            margin *= self.C.DIVISION_GAME_DAMPEN
        shrink = self._environment_shrink(roof)
        if shrink:
            margin *= (1.0 - shrink)
        return margin, {
            "base": base, "hfa": hfa, "rest": rest, "pressure": press,
            "down_leverage": lev, "coach": coach, "qb": qb,
            "qb_fault_sack": qb_fault, "blitz": blitz, "injuries": injuries,
        }

    # ---------------------------------------------------------------- display
    def rating_table(self):
        rows = []
        for team, s in self.teams.items():
            n_p, n_r = self._play_mix(team)
            rows.append({
                "team": team,
                "elo": C.ELO_BASE + C.ELO_PER_POINT * self.team_points(team),
                "power_rating": self.team_points(team),
                "pass_off_pts": n_p * s.pass_off,
                "rush_off_pts": n_r * s.rush_off,
                "pass_def_pts": -n_p * s.pass_def,
                "rush_def_pts": -n_r * s.rush_def,
                "pass_off_epa": s.pass_off,
                "rush_off_epa": s.rush_off,
                "pass_def_epa": s.pass_def,
                "rush_def_epa": s.rush_def,
                "pressure_gen": s.press_def,
                "pressure_allowed": s.press_off,
                "qb_fault_sack_rate": self.qbs.fault_sack_rate(s.qb_recent)
                    if self.qbs.fault_sack_rate(s.qb_recent) is not None else s.qb_fault_sack,
                "blitz_faced": s.blitz_faced,
                "blitz_generated": s.blitz_generated,
                "qb": s.qb_recent,
                "qb_rating": self.qbs.rating(s.qb_recent),
                "games": s.games,
            })
        df = pd.DataFrame(rows).sort_values("elo", ascending=False)
        df.insert(0, "rank", range(1, len(df) + 1))
        return df.reset_index(drop=True)


def run(units: pd.DataFrame, games: pd.DataFrame, through=None, cfg=C):
    """Walk the schedule in order, updating ratings. Returns (model, history).

    `history` holds the PREGAME prediction for every game, which is what you
    backtest on -- never score a model on games it has already absorbed.
    """
    model = Model(cfg)
    # dict lookup instead of .loc -- the fitter calls this hundreds of times
    u = {(r.game_id, r.team): r for r in units.itertuples(index=False)}
    games = games.sort_values(["season", "week", "gameday"])
    g = games[games["result"].notna()]
    if through is not None:
        season, week = through
        g = g[(g.season < season) | ((g.season == season) & (g.week < week))]

    history, cur_season = [], None
    for gm in g.itertuples(index=False):
        if cur_season is not None and gm.season != cur_season:
            for st in model.teams.values():
                st.new_season()
        cur_season = gm.season

        off = u.get((gm.game_id, gm.home_team))
        dfn = u.get((gm.game_id, gm.away_team))
        if off is None or dfn is None:
            continue
        pred, parts = model.predict(
            gm.home_team, gm.away_team,
            neutral=(str(gm.location).lower() == "neutral"),
            home_rest=gm.home_rest, away_rest=gm.away_rest,
            div_game=bool(gm.div_game),
            home_qb=gm.home_qb_id, away_qb=gm.away_qb_id,
            roof=getattr(gm, "roof", None))
            # NOTE: injuries are intentionally not backtested here -- aligning
            # historical injury reports to every past game is unbuilt; the
            # injury term only fires from the CLI for the upcoming week.
        history.append({
            "game_id": gm.game_id, "season": gm.season, "week": gm.week,
            "home": gm.home_team, "away": gm.away_team,
            "pred_margin": pred, "actual_margin": gm.result,
            "market_spread": gm.spread_line, **parts,
        })

        model.observe_game(off, dfn)

    return model, pd.DataFrame(history)
