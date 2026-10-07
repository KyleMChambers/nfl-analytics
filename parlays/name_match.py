"""
name_match.py — The Odds API returns full player names ("Jahmyr Gibbs"),
while nflverse play-by-play uses abbreviated names ("J.Gibbs"). These
won't join on equality, so normalize both to a common key: lowercased
first-initial + last name, with common suffixes stripped.

This is a heuristic, not a guaranteed-unique key -- two players with the
same first initial and last name (e.g. two different "J.Smith"s) will
collide. Given normal NFL rosters this is rare for the specific subset
of players who show up in both red-zone usage data and ATD markets, but
you should spot-check the merged output the first few times you run
this, especially for common surnames.
"""

from __future__ import annotations
import re

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """
    'Jahmyr Gibbs' -> 'j.gibbs'
    'J.Gibbs' -> 'j.gibbs'
    'Kenneth Walker III' -> 'k.walker'
    """
    name = name.strip().lower()
    name = re.sub(r"[^\w\s.]", "", name)  # drop punctuation except periods (keeps "J.Gibbs" intact)
    parts = [p for p in re.split(r"[\s.]+", name) if p and p not in SUFFIXES]

    if not parts:
        return ""
    first_initial = parts[0][0]
    last_name = parts[-1]
    return f"{first_initial}.{last_name}"


def normalize_full_name(name: str) -> str:
    """
    Strict full-name match, for use once BOTH sides of a join have real
    full names (not the lossy abbreviated form). Lowercases, strips
    punctuation/suffixes, keeps every name part -- so "Kyren Williams"
    and "Kyle Williams" stay distinct, unlike normalize_name()'s
    first-initial shortcut which collapses both to "k.williams".

    'Kyren Williams' -> 'kyren williams'
    "Ke'Shawn Williams" -> 'keshawn williams'
    'D.J. Moore' -> 'dj moore'
    'DJ Moore' -> 'dj moore'
    'Kenneth Walker III' -> 'kenneth walker'
    """
    name = name.strip().lower()
    name = re.sub(r"[^\w\s]", "", name)  # drop ALL punctuation (periods, apostrophes, hyphens)
    parts = [p for p in name.split() if p and p not in SUFFIXES]
    return " ".join(parts)


if __name__ == "__main__":
    examples = ["Jahmyr Gibbs", "J.Gibbs", "Kenneth Walker III", "K.Walker", "D.J. Moore", "DJ Moore"]
    for e in examples:
        print(f"{e!r:25s} -> {normalize_name(e)!r}")

    print("\nFull-name matching (fixes the Williams collision):")
    collision_examples = ["Kyren Williams", "Kyle Williams", "Ke'Shawn Williams"]
    for e in collision_examples:
        print(f"{e!r:25s} -> normalize_name={normalize_name(e)!r}  normalize_full_name={normalize_full_name(e)!r}")
