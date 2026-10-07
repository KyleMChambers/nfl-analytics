"""
td_model.py — Turns red-zone usage + historical conversion rates into a
per-player anytime-TD probability for an upcoming week.

Approach: expected TDs (lambda) = expected touches this week * TD rate
per touch * opponent adjustment. Then P(at least one TD) = 1 - e^(-lambda),
treating touchdown events as approximately Poisson over a player's
touches in a game. This is the standard, well-validated approach used
for anytime-TD-scorer modeling -- simple, and it behaves sensibly at
both ends (low-volume players get low but nonzero probability, bellcow
backs get high but still <100% probability).

Two things do most of the work here:
  1. Shrinkage: 2-3 games of red-zone touches is a small sample. A back
     who scored on 2 of 3 red-zone carries isn't a 67% red-zone
     finisher -- shrink that rate toward the league baseline based on
     how many touches they actually have.
  2. A small non-red-zone component, so this doesn't zero out big-play
     receivers/backs who score from outside the 20 more than average.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

RUSH_SHRINKAGE_K = 8    # pseudo-touches added toward baseline for rushing
REC_SHRINKAGE_K = 6     # pass-game TD rates are noisier per-target, slightly lower k needed less smoothing here trade-off
NON_RZ_TD_RATE = 0.02   # flat league-average rate for TDs scored on non-red-zone touches (long runs/passes)


@dataclass
class PlayerTDProjection:
    player_id: str
    player_name: str
    team: str
    opponent: str
    expected_rz_touches: float
    lambda_total: float
    model_prob: float


def shrink_rate(events: float, touches: float, baseline_rate: float, k: float) -> float:
    """Empirical Bayes shrinkage: pulls low-sample rates toward the baseline."""
    return (events + k * baseline_rate) / (touches + k) if (touches + k) > 0 else baseline_rate


def project_player_td_prob(
    rz_carries_per_game: float,
    rz_rush_tds: float,
    rz_carries: float,
    rz_targets_per_game: float,
    rz_rec_tds: float,
    rz_targets: float,
    non_rz_touches_per_game: float,
    opponent_factor: float,
    baseline_rush_rate: float,
    baseline_rec_rate: float,
    open_field_lambda: float | None = None,
) -> tuple[float, float]:
    """
    Returns (lambda_total, model_prob) for one player for one upcoming game.
    non_rz_touches_per_game covers non-red-zone rushes + targets combined,
    priced at a flat small league-average TD rate (long touchdowns are much
    less predictable from usage alone, so this stays simple by design).
    """
    shrunk_rush_rate = shrink_rate(rz_rush_tds, rz_carries, baseline_rush_rate, RUSH_SHRINKAGE_K)
    shrunk_rec_rate = shrink_rate(rz_rec_tds, rz_targets, baseline_rec_rate, REC_SHRINKAGE_K)

    lambda_rush = rz_carries_per_game * shrunk_rush_rate * opponent_factor
    lambda_rec = rz_targets_per_game * shrunk_rec_rate * opponent_factor
    # Open-field TDs: use the player's real open-field volume when we have
    # it (open_field_lambda), otherwise fall back to the old flat guess.
    if open_field_lambda is not None:
        lambda_other = open_field_lambda
    else:
        lambda_other = non_rz_touches_per_game * NON_RZ_TD_RATE

    lambda_total = lambda_rush + lambda_rec + lambda_other
    model_prob = 1 - np.exp(-lambda_total)

    return lambda_total, model_prob


def build_weekly_projections(
    usage: pd.DataFrame,
    baselines: dict,
    opponent_factors: pd.DataFrame,
    matchups: dict[str, str],   # {team_abbrev: opponent_abbrev} for the upcoming week
    team_lookup: dict[str, str],  # {player_id: team_abbrev} -- current-week roster/team mapping
    non_rz_touches_lookup: dict[str, float] | None = None,
    min_games_played: int = 2,
    prior_season_usage: pd.DataFrame | None = None,
    prior_weight_games: float = 6.0,
    full_name_lookup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    matchups and team_lookup need to reflect the CURRENT week's slate and
    rosters (a player's team can change via trade/signing) -- pull these
    fresh each week rather than reusing last week's mapping.

    full_name_lookup ({player_id: "Kyren Williams"}, from the roster
    file) replaces the output player_name with the real full name when
    available, instead of nflverse play-by-play's abbreviated form
    ("K.Williams"). This matters: abbreviated names collide across
    different real players (confirmed in testing -- Kyren Williams, Kyle
    Williams, and Ke'Shawn Williams all reduce to the same "K.Williams"),
    so matching against the Odds API's full names needs the real full
    name on this side too, not the lossy abbreviation. Falls back to the
    abbreviated pbp name only if a player_id isn't found on the current
    roster (rare -- e.g. just-released player still in historical data).

    Early in a season, EVERY player's current-season red-zone sample is
    small -- 2-3 games isn't enough to tell an established role from a
    fluke, for anyone. min_games_played alone doesn't fix this (a real
    problem seen in testing: legitimate players with genuinely low
    current-season volume, and backups with a one-off flukey touch, both
    look similar on 2-3 games of same-season-only data).

    prior_season_usage (pass last season's compute_player_red_zone_usage
    output) fixes this by blending: a player's effective volume becomes a
    weighted average of this season's rate and last season's rate, with
    prior_weight_games acting as "how many games worth of trust to put in
    last season's role." More current-season games played pulls the
    estimate toward this season; fewer leans on last season. A rookie
    with no prior-season row gets treated as replacement-level volume
    until they build a current-season sample -- a deliberate conservative
    default, not an oversight.
    """
    non_rz_touches_lookup = non_rz_touches_lookup or {}
    full_name_lookup = full_name_lookup or {}
    rows = []

    prior_lookup = {}
    if prior_season_usage is not None:
        for _, prow in prior_season_usage.iterrows():
            prior_lookup[prow["player_id"]] = prow

    for _, row in usage.iterrows():
        if row["games_played"] < min_games_played:
            continue

        team = team_lookup.get(row["player_id"])
        if team is None or team not in matchups:
            continue  # player's team not in this week's slate (bye week, etc.)
        opponent = matchups[team]

        opp_row = opponent_factors[opponent_factors["defteam"] == opponent]
        opponent_factor = float(opp_row["opponent_factor"].iloc[0]) if len(opp_row) else 1.0

        non_rz_touches = non_rz_touches_lookup.get(row["player_id"], 2.0)  # conservative default

        prior = prior_lookup.get(row["player_id"])
        games_played = row["games_played"]

        # Real open-field volume, blended with last season like the red-zone
        # numbers. Long TDs are mostly luck per play, so priced at the league
        # rate rather than the player's own (tiny-sample) open-field TD rate.
        open_field_lambda = None
        if "or_carries" in row.index and "or_rush_td_rate" in baselines:
            p_orc = prior["or_carries_per_game"] if (prior is not None and "or_carries_per_game" in prior.index) else 0.0
            p_ort = prior["or_targets_per_game"] if (prior is not None and "or_targets_per_game" in prior.index) else 0.0
            w = prior_weight_games if prior is not None else 0.0
            orc_pg = (row["or_carries"] + w * p_orc) / (games_played + w)
            ort_pg = (row["or_targets"] + w * p_ort) / (games_played + w)
            open_field_lambda = orc_pg * baselines["or_rush_td_rate"] + ort_pg * baselines["or_rec_td_rate"]

        if prior is not None:
            # Blend current-season per-game volume with last season's, weighted
            # by how many current-season games we've actually seen.
            blended_rz_carries_pg = (
                row["rz_carries"] + prior_weight_games * prior["rz_carries_per_game"]
            ) / (games_played + prior_weight_games)
            blended_rz_targets_pg = (
                row["rz_targets"] + prior_weight_games * prior["rz_targets_per_game"]
            ) / (games_played + prior_weight_games)
            # Pool raw touch/TD counts across both seasons for the rate
            # shrinkage in project_player_td_prob -- more total sample
            # means less reliance on the league-baseline fallback.
            pooled_rz_carries = row["rz_carries"] + prior["rz_carries"]
            pooled_rz_rush_tds = row["rz_rush_tds"] + prior["rz_rush_tds"]
            pooled_rz_targets = row["rz_targets"] + prior["rz_targets"]
            pooled_rz_rec_tds = row["rz_rec_tds"] + prior["rz_rec_tds"]
        else:
            # No prior-season row (true rookie, or didn't qualify last season) --
            # stay conservative: blend current season toward zero volume rather
            # than assuming a league-average role they haven't earned yet.
            blended_rz_carries_pg = (row["rz_carries"]) / (games_played + prior_weight_games)
            blended_rz_targets_pg = (row["rz_targets"]) / (games_played + prior_weight_games)
            pooled_rz_carries = row["rz_carries"]
            pooled_rz_rush_tds = row["rz_rush_tds"]
            pooled_rz_targets = row["rz_targets"]
            pooled_rz_rec_tds = row["rz_rec_tds"]

        lam, prob = project_player_td_prob(
            rz_carries_per_game=blended_rz_carries_pg,
            rz_rush_tds=pooled_rz_rush_tds,
            rz_carries=pooled_rz_carries,
            rz_targets_per_game=blended_rz_targets_pg,
            rz_rec_tds=pooled_rz_rec_tds,
            rz_targets=pooled_rz_targets,
            non_rz_touches_per_game=non_rz_touches,
            opponent_factor=opponent_factor,
            baseline_rush_rate=baselines["rush_td_rate"],
            baseline_rec_rate=baselines["rec_td_rate"],
            open_field_lambda=open_field_lambda,
        )

        rows.append({
            "player_id": row["player_id"],
            "player_name": full_name_lookup.get(row["player_id"], row["player_name"]),
            "team": team,
            "opponent": opponent,
            "expected_rz_touches": blended_rz_carries_pg + blended_rz_targets_pg,
            "has_prior_season": prior is not None,
            "lambda_total": lam,
            "model_prob": prob,
        })

    return pd.DataFrame(rows).sort_values("model_prob", ascending=False)


if __name__ == "__main__":
    from data_loader import load_pbp, compute_player_red_zone_usage, league_baseline_rates, team_red_zone_defense

    pbp = load_pbp(2026)
    usage = compute_player_red_zone_usage(pbp)
    baselines = league_baseline_rates(usage)
    opp_factors = team_red_zone_defense(pbp)

    # Minimal example matchup for week 4 -- in the real weekly pipeline this
    # comes from the current week's schedule, pulled fresh (see weekly_pipeline.py).
    example_matchups = {"DET": "CIN", "CIN": "DET", "BAL": "HOU", "HOU": "BAL"}
    team_lookup = {
        pid: "DET" for pid in usage[usage["player_name"] == "J.Gibbs"]["player_id"]
    }
    team_lookup.update({
        pid: "BAL" for pid in usage[usage["player_name"] == "D.Henry"]["player_id"]
    })

    proj = build_weekly_projections(usage, baselines, opp_factors, example_matchups, team_lookup)
    print(proj.to_string(index=False))
