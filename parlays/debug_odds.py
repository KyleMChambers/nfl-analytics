"""
debug_odds.py — One-off diagnostic. Prints the RAW response from The
Odds API for a single game's anytime-TD market, with no parsing or
interpretation applied. Run this, paste the output back, and we can see
exactly what shape the data is in rather than guessing at it.

Usage:
    python debug_odds.py
(ODDS_API_KEY must already be set in your terminal session, same as
for weekly_pipeline.py)
"""

import os
import json
import requests

ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")
EVENTS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events"
EVENT_ODDS_URL_TMPL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/events/{event_id}/odds"

if not ODDS_API_KEY:
    raise RuntimeError("Set ODDS_API_KEY first, same as before.")

print("Fetching event list...")
events_resp = requests.get(EVENTS_URL, params={"apiKey": ODDS_API_KEY})
events_resp.raise_for_status()
events = events_resp.json()

# Find the Bengals @ Jaguars game specifically (Chase Brown's team),
# since that's one of the suspicious legs. Falls back to the first
# event if that matchup isn't found.
target = None
for e in events:
    if "Bengals" in e.get("home_team", "") or "Bengals" in e.get("away_team", ""):
        target = e
        break
if target is None:
    target = events[0]
    print(f"Bengals game not found by name -- using first event instead: {target['away_team']} @ {target['home_team']}")
else:
    print(f"Found: {target['away_team']} @ {target['home_team']}")

print(f"\nFetching player_anytime_td market for event {target['id']}...\n")
resp = requests.get(
    EVENT_ODDS_URL_TMPL.format(event_id=target["id"]),
    params={
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "player_anytime_td",
        "oddsFormat": "american",
        "bookmakers": "fanduel,betmgm",
    },
)
resp.raise_for_status()
data = resp.json()

print("=" * 70)
print("RAW RESPONSE (full, unparsed):")
print("=" * 70)
print(json.dumps(data, indent=2))
