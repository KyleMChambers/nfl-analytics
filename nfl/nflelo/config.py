"""Tunable parameters for the NFL unit-Elo model.

All ratings live in EPA-per-play space (i.e. points per play above average).
Elo-style display numbers are derived at the end: ELO = 1500 + 25 * (points vs avg).
"""

# ---------------------------------------------------------------- data sources
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"
GAMES_URL = "https://github.com/nflverse/nfldata/raw/master/data/games.csv"
PFR_DEF_URL = "https://www.pro-football-reference.com/years/{season}/defense.htm"

# ------------------------------------------------------------- learning rates
# K = fraction of a single game's surprise absorbed into the rating.
# Offense is more stable than defense, so it earns more of the credit.
K_PASS_OFF = 0.05
K_PASS_DEF = 0.015
K_RUSH_OFF = 0.075
K_RUSH_DEF = 0.05

# Early-season K multiplier: ratings move faster before the sample stabilizes.
EARLY_K_MULT = 1.5
EARLY_K_DECAY_GAMES = 6.0  # multiplier decays 2.0 -> 1.0 over this many games

# Offseason carryover. 1.0 = no regression, 0.0 = everyone resets to average.
CARRYOVER_PASS = 0.62
CARRYOVER_RUSH = 0.5  # rushing is noisier year to year; regress it harder

# --------------------------------------------------------------- play volumes
# League-average team-game play counts, used to convert EPA/play into points.
DROPBACKS_PER_GAME = 37.0
RUSHES_PER_GAME = 25.0
# How much of a team's own pass/run tendency to trust vs league average.
# Play mix is endogenous to game script, so it gets shrunk hard.
PLAY_MIX_WEIGHT = 0.0

# ------------------------------------------------------------- game modifiers
HOME_FIELD_POINTS = 1.5   # post-2020 reality, not the folk-wisdom 3
REST_POINTS_PER_DAY = 0.09  # applied to (rest_diff), capped
REST_DIFF_CAP = 7
DIVISION_GAME_DAMPEN = 1.0  # familiarity compresses margins slightly

# Indoor/controlled environments have less weather variance to swing a game
# either direction; applied as a shrink on |margin|, not a points add -- a
# dome doesn't make either team better, it makes the game hew closer to the
# neutral-field baseline the ratings already assume. Tested the same way as
# everything else above: the improvement on 2024-2026 holdout was inside
# noise (13.0624 -> 13.0622 RMSE), so it's off by default. Small positive
# values are defensible on priors even without a measurable backtest gain;
# raise this if you want to carry it anyway.
DOME_MARGIN_SHRINK = 0.0   # closed/dome roof: shrink |margin| by this fraction

# ------------------------------------------------------------ injury layer
# Approximate, NOT fitted -- there isn't enough per-position injury sample to
# fit these safely. Treat as an informed prior, tune by feel, revisit once a
# season or two of --injuries backtests exist. Point value of losing a
# FULL-SNAP-SHARE starter at each position, scaled by actual snap share and
# report severity.
INJURY_POSITION_VALUE = {
    "QB": 0.0,   # handled separately by the QB rating, not here
    "RB": 0.6, "WR": 0.7, "TE": 0.35,
    "T": 0.55, "G": 0.35, "C": 0.4,
    "EDGE": 0.55, "DE": 0.5, "DT": 0.3, "NT": 0.25,
    "LB": 0.3, "ILB": 0.3, "OLB": 0.4,
    "CB": 0.5, "S": 0.3, "FS": 0.3, "SS": 0.3,
    "K": 0.05, "P": 0.02,
}
INJURY_STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.30}
INJURY_SNAP_LOOKBACK_WEEKS = 4

# ------------------------------------------------------- metric adjustment weights
# Each term is a shrunk, opponent-agnostic residual signal that EPA alone
# either misses (pressure) or overstates (turnover luck, 3rd-down spikes).
W_PRESSURE = 0.1        # points per 1pp of pressure-rate edge, per game
W_THIRD_DOWN = 0.2      # points per unit of down-leverage residual
W_TURNOVER_LUCK = 0.5   # 1.0 = fully strip out recovery luck from ratings
W_COACH = 0.0           # OFF by default: coaching behavior did not generalize
                        # out of sample (see README). Set ~0.3 to experiment.
W_QB = 0.25              # 1.0 = apply full QB delta on a starter change

# FTN charting has no direct pressure flag -- it refines sack attribution
# and exposes blitz rate. Both terms are OFF by default: fit on 2018-2023,
# scored on 2024-2026 held out, neither improved RMSE (same test that kept
# pressure and QB identity in and coaching record out -- see README). The
# columns still appear in spreads output at 0.0 so the plumbing is visible;
# raise these if you want to keep experimenting on a longer sample.
W_QB_FAULT_SACK = 0.0   # points per unit of qb-fault-sack-rate edge
W_BLITZ = 0.0           # points per unit of blitz-rate mismatch

POINTS_PER_TURNOVER = 3.6  # swing value of one takeaway
SACK_POINTS = 1.75         # EPA cost of one sack, used for QB sack-rate carry

# Shrinkage: stat is pulled toward league mean by n/(n+STABILIZE_N).
STABILIZE = {
    "pressure": 4.0,      # pressure rate stabilizes fast
    "third_down": 10.0,   # 3rd-down rates regress hard
    "turnover": 8.0,
    "coach": 6.0,
    "qb": 120.0,          # dropbacks, not games
}

# ------------------------------------------------------------------- display
ELO_BASE = 1500.0
ELO_PER_POINT = 25.0
