"""
parlay_builder.py — Combines positive-edge anytime-TD legs into ranked,
sized parlays for the week.

Constraint that matters here: ONE LEG PER GAME. Two ATD legs from the
same game are correlated (a high-scoring game lifts both players'
chances together), and sportsbooks already price that correlation into
same-game parlay (SGP) odds -- often less favorably than the naive
multiplication this script does. Restricting to one leg per game keeps
every combo close to genuinely independent, so the combined odds you
see are a fair reflection of the combined probability, not something
the book would reprice against you if built as an SGP.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations


@dataclass
class ATDLeg:
    player_name: str
    player_id: str
    team: str
    opponent: str
    game_id: str
    model_prob: float
    market_prob: float
    yes_odds_american: int
    yes_odds_decimal: float
    book: str
    prop: str = "Anytime TD"   # e.g. "Anytime TD", "100+ Rush Yds"
    our_model: float | None = None   # our model's chance (before blending)
    consensus: float | None = None   # other books' average chance, cut removed
    n_books: int = 0                 # how many other books priced it

    def summary(self) -> str:
        m = f"Model {self.our_model:.0%}" if self.our_model is not None else "Model  --"
        k = f"Mkt {self.consensus:.0%} ({self.n_books} bks)" if self.consensus is not None else "Mkt  --        "
        return f"BetMGM {self.market_prob:.0%} | {k} | {m} | Final {self.model_prob:.0%}"

    @property
    def edge(self) -> float:
        return self.model_prob - self.market_prob

    @property
    def edge_pct(self) -> float:
        return self.edge / self.market_prob if self.market_prob > 0 else float("nan")


@dataclass
class ATDParlay:
    legs: list[ATDLeg]
    combined_model_prob: float
    combined_decimal_odds: float
    implied_fair_prob: float
    edge_pct: float
    suggested_stake: float
    potential_payout: float

    def describe(self) -> str:
        leg_strs = [f"    {l.player_name} {l.prop} ({l.team} vs {l.opponent}, {l.yes_odds_american:+d}, {l.book})" for l in self.legs]
        american = round((self.combined_decimal_odds - 1) * 100)
        one_in = 1 / self.combined_model_prob if self.combined_model_prob > 0 else float("inf")
        return (
            "\n".join(leg_strs) + "\n"
            f"  Parlay odds: +{american:,}  |  Model chance all hit: {self.combined_model_prob:.2%} (about 1 in {one_in:,.0f})"
            f"  |  Edge: {self.edge_pct:+.1%}\n"
            f"  Bet ${self.suggested_stake:.2f} -> pays ${self.potential_payout:,.2f}"
        )


def build_parlay(legs: list[ATDLeg], bankroll: float, kelly_frac: float = 0.10, fixed_stake: float | None = None) -> ATDParlay:
    """
    kelly_frac defaults even lower here (0.10) than the futures parlay
    project (0.15) because: (a) these are 3-8+ leg combos, so variance
    compounds harder, and (b) the model's edge estimate itself is less
    mature -- red-zone usage models need real backtesting before you'd
    want to size these anywhere close to full Kelly.
    """
    combined_prob = 1.0
    combined_odds = 1.0
    for leg in legs:
        combined_prob *= leg.model_prob
        combined_odds *= leg.yes_odds_decimal

    implied_fair_prob = 1 / combined_odds
    edge_pct = (combined_prob - implied_fair_prob) / implied_fair_prob if implied_fair_prob > 0 else float("nan")

    b = combined_odds - 1
    p = combined_prob
    q = 1 - p
    f_star = (b * p - q) / b if b > 0 else 0
    f_star = max(f_star, 0) * kelly_frac

    stake = round(min(f_star * bankroll, 20.0), 2)  # hard cap at $20 per your stated stake range
    stake = max(stake, 1.0) if f_star > 0 else 0.0
    if fixed_stake is not None:
        stake = round(fixed_stake, 2)
    payout = round(stake * combined_odds, 2)

    return ATDParlay(
        legs=legs,
        combined_model_prob=combined_prob,
        combined_decimal_odds=combined_odds,
        implied_fair_prob=implied_fair_prob,
        edge_pct=edge_pct,
        suggested_stake=stake,
        potential_payout=payout,
    )


def rank_weekly_parlays(
    available_legs: list[ATDLeg],
    bankroll: float,
    min_legs: int = 4,
    max_legs: int = 6,
    top_n: int = 15,
    min_edge_pct: float = 0.05,
    max_candidate_legs: int = 25,
    stake: float = 10.0,
    min_payout: float = 1_000,
    max_payout: float = 100_000,
    sort_by: str = "hit_chance",
    max_per_game: int = 3,
) -> list[ATDParlay]:
    """
    Builds every combo of min_legs..max_legs positive-edge legs (one per
    game, one per player), keeps only the ones whose payout on `stake`
    lands between min_payout and max_payout, then ranks them.

    sort_by:
      "hit_chance" -> most likely to hit first (default)
      "edge"       -> biggest model-vs-book disagreement first

    max_candidate_legs caps the pool before combining: 6 legs from 25
    is ~177k combos (instant); 6 from 190 is 30+ billion (looks frozen).
    """
    positive_legs = [l for l in available_legs if l.edge_pct > min_edge_pct]
    positive_legs.sort(key=lambda l: l.edge_pct, reverse=True)
    if len(positive_legs) > max_candidate_legs:
        print(f"  [note] {len(positive_legs)} legs cleared {min_edge_pct:.0%} edge -- "
              f"using the top {max_candidate_legs} by edge (max {max_per_game} per game) to keep it fast.")
        pool, per_game = [], {}
        for l in positive_legs:
            if per_game.get(l.game_id, 0) >= max_per_game:
                continue  # keep the pool spread across games, not 10 legs from one game
            pool.append(l)
            per_game[l.game_id] = per_game.get(l.game_id, 0) + 1
            if len(pool) >= max_candidate_legs:
                break
        positive_legs = pool

    all_parlays: list[ATDParlay] = []
    for n in range(min_legs, max_legs + 1):
        for combo in combinations(positive_legs, n):
            if len({l.game_id for l in combo}) != n:
                continue  # one leg per game
            if len({l.player_id for l in combo}) != n:
                continue  # never the same player twice
            odds = 1.0
            for l in combo:
                odds *= l.yes_odds_decimal
            payout = stake * odds
            if not (min_payout <= payout <= max_payout):
                continue  # outside your payout window
            all_parlays.append(build_parlay(list(combo), bankroll, fixed_stake=stake))

    if sort_by == "edge":
        all_parlays.sort(key=lambda p: p.edge_pct, reverse=True)
    else:
        all_parlays.sort(key=lambda p: p.combined_model_prob, reverse=True)

    # Avoid 15 near-identical parlays: limit how often one player can repeat.
    picked, use_count = [], {}
    for p in all_parlays:
        if any(use_count.get(l.player_id, 0) >= 3 for l in p.legs):
            continue
        picked.append(p)
        for l in p.legs:
            use_count[l.player_id] = use_count.get(l.player_id, 0) + 1
        if len(picked) >= top_n:
            break
    return picked


if __name__ == "__main__":
    # Example shortlist -- in the real pipeline this comes from joining
    # td_model.py's projections against devig_props.py's market prices.
    legs = [
        ATDLeg("J.Gibbs", "p1", "DET", "CIN", "2026_04_DET_CIN", model_prob=0.77, market_prob=0.336, yes_odds_american=180, yes_odds_decimal=2.80, book="fanduel"),
        ATDLeg("D.Henry", "p2", "BAL", "HOU", "2026_04_BAL_HOU", model_prob=0.55, market_prob=0.371, yes_odds_american=145, yes_odds_decimal=2.45, book="betmgm"),
        ATDLeg("A.Jeanty", "p3", "LV", "IND", "2026_04_LV_IND", model_prob=0.40, market_prob=0.290, yes_odds_american=200, yes_odds_decimal=3.00, book="fanduel"),
        ATDLeg("K.Walker", "p4", "SEA", "ARI", "2026_04_SEA_ARI", model_prob=0.50, market_prob=0.400, yes_odds_american=130, yes_odds_decimal=2.30, book="betmgm"),
        ATDLeg("J.Taylor", "p5", "IND", "LV", "2026_04_LV_IND", model_prob=0.45, market_prob=0.350, yes_odds_american=155, yes_odds_decimal=2.55, book="fanduel"),
    ]

    bankroll = 2000
    top = rank_weekly_parlays(legs, bankroll, min_legs=3, max_legs=4, top_n=5, min_payout=10, max_payout=10_000)

    print(f"Top weekly ATD parlays (bankroll=${bankroll}):\n")
    for i, p in enumerate(top, 1):
        print(f"#{i}")
        print(p.describe())
        print()
