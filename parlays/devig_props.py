"""
devig_props.py — Anytime TD is a two-way market (Yes / No), unlike the
multi-way futures market from the old project. Devigging a two-way
market is simpler: convert each side's American odds to implied
probability, then normalize so they sum to 1.

Sportsbooks typically don't publish the "No" side for every ATD prop
(FanDuel/BetMGM sometimes only price the "Yes"). When only "Yes" is
available, this module falls back to a standard assumed hold based on
the book's typical player-prop vig, documented below -- less precise
than a true two-sided devig, so real "No" prices should be used
whenever the book offers them.
"""

from __future__ import annotations
from dataclasses import dataclass

# Typical total vig FanDuel/BetMGM carry on player TD props when only
# the "Yes" side is quoted. This is an approximation for the fallback
# path only -- always prefer a real two-sided price when available.
ASSUMED_TOTAL_VIG = 0.10


def american_to_prob(odds: int) -> float:
    """Convert American odds to implied probability (includes vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1 + odds / 100
    else:
        return 1 + 100 / -odds


@dataclass
class PropPrice:
    player_name: str
    yes_odds: int              # American odds, e.g. -150 or +220
    no_odds: int | None = None  # None if book doesn't quote a "No" price


def devig_two_way(price: PropPrice) -> float:
    """
    Returns the de-vigged "Yes" (anytime TD) probability.
    If a real "No" price is available, normalize both sides against each
    other -- this is the accurate path and should be used whenever possible.
    If not, back out vig using the assumed total vig constant above.
    """
    yes_implied = american_to_prob(price.yes_odds)

    if price.no_odds is not None:
        no_implied = american_to_prob(price.no_odds)
        total = yes_implied + no_implied
        return yes_implied / total

    # Fallback: assume the "Yes" price alone carries half the total vig
    # (a reasonable approximation for how books typically distribute it
    # on short-priced favorites vs. longshots in this market).
    return yes_implied / (1 + ASSUMED_TOTAL_VIG)


def compute_edge(model_prob: float, market_prob: float) -> dict:
    edge = model_prob - market_prob
    edge_pct = edge / market_prob if market_prob > 0 else float("nan")
    return {"model_prob": model_prob, "market_prob": market_prob, "edge": edge, "edge_pct": edge_pct}


if __name__ == "__main__":
    # Example: a player priced at +180 to score anytime, book also quotes
    # the "No" side at -240.
    price = PropPrice(player_name="J.Gibbs", yes_odds=180, no_odds=-240)
    market_prob = devig_two_way(price)
    print(f"Raw 'Yes' implied prob: {american_to_prob(price.yes_odds):.3%}")
    print(f"De-vigged market prob:  {market_prob:.3%}")

    model_prob = 0.77  # from td_model.py's projection for Gibbs above
    edge = compute_edge(model_prob, market_prob)
    print(f"\nModel vs. market: {edge}")

    # Example with no "No" price quoted -- fallback path
    price2 = PropPrice(player_name="D.Henry", yes_odds=145, no_odds=None)
    market_prob2 = devig_two_way(price2)
    print(f"\nD.Henry (fallback devig, no 'No' price quoted): {market_prob2:.3%}")
